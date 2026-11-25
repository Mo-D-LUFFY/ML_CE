"""Milestone 2: PyTorch MLP on the official Adult split; validation early stopping."""
import argparse
import json
import time
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.model_selection import train_test_split
from train_classical import ROOT, ALL, SIX, load, make_model, scores


def build_mlp(input_dim):
    return nn.Sequential(nn.Linear(input_dim, 128), nn.LayerNorm(128), nn.ReLU(),
        nn.Dropout(.2), nn.Linear(128, 64), nn.ReLU(), nn.Dropout(.2),
        nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1))


def seed_everything(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def encode(pre, x, device):
    a = pre.transform(x)
    if hasattr(a, 'toarray'):
        a = a.toarray()
    return torch.as_tensor(np.asarray(a, dtype=np.float32), device=device)


def labels(y, device):
    return torch.as_tensor(y.to_numpy(dtype=np.float32), device=device).view(-1, 1)


@torch.inference_mode()
def predict(net, x):
    net.eval()
    return torch.cat([net(b).sigmoid().flatten().cpu() for b in x.split(4096)]).numpy()


def train_epoch(net, x, y, opt, batch_size):
    net.train()
    total = torch.zeros((), device=x.device)
    for idx in torch.randperm(len(x), device=x.device).split(batch_size):
        opt.zero_grad(set_to_none=True)
        loss = nn.functional.binary_cross_entropy_with_logits(net(x[idx]), y[idx])
        loss.backward()
        opt.step()
        total += loss.detach() * len(idx)
    return float(total.cpu()) / len(x)


class MLPPredictor:
    """Adapter for later counterfactual search; accepts original feature DataFrames."""
    def __init__(self, directory, tag='full', device='cpu'):
        directory = Path(directory)
        self.pre = joblib.load(directory / f'{tag}_preprocessor.joblib')
        ckpt = torch.load(directory / f'{tag}_mlp.pt', map_location=device, weights_only=True)
        self.features = ckpt['features']
        self.device = torch.device(device)
        self.net = build_mlp(ckpt['input_dim']).to(self.device)
        self.net.load_state_dict(ckpt['state_dict'])

    def predict_proba(self, x):
        p = predict(self.net, encode(self.pre, x[self.features], self.device))
        return np.column_stack((1-p, p))

    def predict(self, x):
        return (self.predict_proba(x)[:, 1] >= .5).astype(int)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data-dir', type=Path, default=ROOT.parent/'cs725_starter'/'data')
    ap.add_argument('--out', type=Path, default=ROOT/'results'/'mlp')
    ap.add_argument('--features', choices=['six','full','both'], default='both')
    ap.add_argument('--device', choices=['auto','cpu','cuda'], default='auto')
    ap.add_argument('--epochs', type=int, default=50)
    ap.add_argument('--patience', type=int, default=7)
    ap.add_argument('--batch-size', type=int, default=512)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--smoke', action='store_true')
    args = ap.parse_args()
    if min(args.epochs,args.patience,args.batch_size) < 1:
        ap.error('epochs, patience, batch size must be positive')
    device = torch.device(('cuda' if torch.cuda.is_available() else 'cpu') if args.device=='auto' else args.device)
    if device.type=='cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; use --device cpu')
    torch.set_num_threads(4)
    print(f'Device: {device}' + (f' ({torch.cuda.get_device_name(0)})' if device.type=='cuda' else ''), flush=True)
    (train,test), hashes = load(args.data_dir,args.smoke)
    y=(train.income=='>50K').astype(int); yt=(test.income=='>50K').astype(int)
    fi,vi=train_test_split(np.arange(len(train)),test_size=.2,stratify=y,random_state=args.seed)
    out=args.out; out.mkdir(parents=True,exist_ok=True)
    rows=[]; history=[]; predictions=pd.DataFrame({'row_id':np.arange(len(test)), 'target':yt})
    tags=['six','full'] if args.features=='both' else [args.features]
    meta=dict(dataset='fixture' if args.smoke else 'UCI Adult official split', n_train=len(train),
        n_validation=len(vi),n_test=len(test), seed=args.seed, threshold=.5,sha256=hashes,
        architecture='128-64-32-1, ReLU, LayerNorm, dropout 0.2', optimizer='AdamW',lr=.001,
        weight_decay=.0001,batch_size=args.batch_size,max_epochs=args.epochs,patience=args.patience,
        torch=str(torch.__version__),device=str(device),feature_sets=tags,
        protocol='Early stopping by validation BCE; fresh refit on all official training rows for selected epochs')
    (out/'metadata.json').write_text(json.dumps(meta,indent=2),encoding='utf8')
    for tag in tags:
        features=SIX if tag=='six' else ALL
        seed_everything(args.seed)
        pre=make_model('lr',features,args.seed,1).steps[0][1]
        pre.fit(train.iloc[fi][features]); x=encode(pre,train.iloc[fi][features],device)
        xv=encode(pre,train.iloc[vi][features],device); yf=labels(y.iloc[fi],device); yv=labels(y.iloc[vi],device)
        net=build_mlp(x.shape[1]).to(device); opt=torch.optim.AdamW(net.parameters(),lr=.001,weight_decay=.0001)
        best=float('inf'); best_epoch=0
        for epoch in range(1,args.epochs+1):
            loss=train_epoch(net,x,yf,opt,args.batch_size)
            net.eval()
            with torch.inference_mode():
                vl=float(nn.functional.binary_cross_entropy_with_logits(net(xv),yv).cpu())
            pv=predict(net,xv); vs=scores(y.iloc[vi],pv)
            history.append(dict(features=tag,epoch=epoch,train_loss=loss,val_loss=vl,val_f1=vs['f1'],val_auc=vs['roc_auc']))
            if epoch==1 or epoch%5==0:
                print(f'{tag} epoch {epoch}: train={loss:.4f} val={vl:.4f} F1={vs["f1"]:.4f}',flush=True)
            if vl < best-1e-5:
                best=vl; best_epoch=epoch; best_scores=vs
            if epoch-best_epoch>=args.patience: break
        rows.append(dict(model='mlp',features=tag,split='validation',n=len(vi),epochs=best_epoch,**best_scores))
        seed_everything(args.seed)
        pre=make_model('lr',features,args.seed,1).steps[0][1]
        pre.fit(train[features]); x=encode(pre,train[features],device); yf=labels(y,device)
        net=build_mlp(x.shape[1]).to(device); opt=torch.optim.AdamW(net.parameters(),lr=.001,weight_decay=.0001)
        start=time.perf_counter()
        for _ in range(best_epoch): train_epoch(net,x,yf,opt,args.batch_size)
        if device.type=='cuda': torch.cuda.synchronize()
        seconds=time.perf_counter()-start
        pt=predict(net,encode(pre,test[features],device)); ts=scores(yt,pt)
        rows.append(dict(model='mlp',features=tag,split='test',n=len(test),epochs=best_epoch,train_seconds=seconds,**ts))
        predictions[f'{tag}_mlp']=pt
        joblib.dump(pre,out/f'{tag}_preprocessor.joblib',compress=3)
        torch.save(dict(features=features,input_dim=x.shape[1],state_dict={k:v.cpu() for k,v in net.state_dict().items()}),out/f'{tag}_mlp.pt')
        pd.DataFrame(rows).to_csv(out/'metrics.csv',index=False)
        pd.DataFrame(history).to_csv(out/'history.csv',index=False)
        predictions.to_csv(out/'test_predictions.csv',index=False)
        restored=MLPPredictor(out,tag,device)
        np.testing.assert_allclose(restored.predict_proba(test[features].iloc[:16])[:,1],pt[:16],rtol=1e-5,atol=1e-6)
        print(f'{tag}_mlp test F1={ts["f1"]:.4f} AUC={ts["roc_auc"]:.4f} accuracy={ts["accuracy"]:.4f} epochs={best_epoch} fit={seconds:.1f}s',flush=True)
    print(f'Results: {out.resolve()}')


if __name__=='__main__': main()
