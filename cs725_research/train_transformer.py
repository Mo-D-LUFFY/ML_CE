"""Compact feature-token Transformer for the official Adult benchmark."""
import argparse
import json
import time
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from train_classical import ROOT, ALL, SIX, NUM, load, scores
from train_mlp import seed_everything, encode, labels, train_epoch


def make_pre(features):
    return ColumnTransformer([
        ('num', make_pipeline(SimpleImputer(strategy='median'), StandardScaler()),
         [f for f in features if f in NUM]),
        ('cat', make_pipeline(SimpleImputer(strategy='constant', fill_value='MISSING'),
            OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)),
         [f for f in features if f not in NUM])])


def config(pre, features):
    enc = pre.named_transformers_['cat'].steps[-1][1]
    return dict(n_num=sum(f in NUM for f in features),
                cards=[len(c)+1 for c in enc.categories_], dim=48, heads=4, layers=2, dropout=.1)


class FTTransformer(nn.Module):
    def __init__(self, n_num, cards, dim=48, heads=4, layers=2, dropout=.1):
        super().__init__()
        self.n_num = n_num
        self.num_weight = nn.Parameter(torch.empty(n_num, dim))
        self.num_bias = nn.Parameter(torch.zeros(n_num, dim))
        nn.init.normal_(self.num_weight, std=.02)
        self.cat_embeddings = nn.ModuleList([nn.Embedding(c, dim) for c in cards])
        self.cls = nn.Parameter(torch.zeros(1, 1, dim))
        layer = nn.TransformerEncoderLayer(d_model=dim, nhead=heads,
            dim_feedforward=dim*2, dropout=dropout, activation='gelu',
            batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=layers, enable_nested_tensor=False)
        self.head = nn.Sequential(nn.LayerNorm(dim), nn.ReLU(), nn.Linear(dim,1))

    def forward(self, x):
        num = x[:, :self.n_num, None] * self.num_weight + self.num_bias
        cats = [emb(x[:, self.n_num+i].long()+1)[:, None, :]
                for i,emb in enumerate(self.cat_embeddings)]
        tokens = torch.cat([self.cls.expand(len(x), -1, -1), num, *cats], dim=1)
        return self.head(self.encoder(tokens)[:,0])


@torch.inference_mode()
def predict(net, x):
    net.eval()
    return torch.cat([net(b).sigmoid().flatten().cpu() for b in x.split(1024)]).numpy()


class FTPredictor:
    def __init__(self, directory, tag='full', device='cpu'):
        directory = Path(directory)
        self.pre = joblib.load(directory / f'{tag}_preprocessor.joblib')
        ckpt = torch.load(directory / f'{tag}_ft.pt', map_location=device, weights_only=True)
        self.features = ckpt['features']
        self.device = torch.device(device)
        self.net = FTTransformer(**ckpt['config']).to(self.device)
        self.net.load_state_dict(ckpt['state_dict'])

    def predict_proba(self, x):
        p = predict(self.net, encode(self.pre, x[self.features], self.device))
        return np.column_stack((1-p,p))

    def predict(self, x):
        return (self.predict_proba(x)[:,1]>=.5).astype(int)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data-dir',type=Path,default=ROOT.parent/'cs725_starter'/'data')
    ap.add_argument('--out',type=Path,default=ROOT/'results'/'transformer')
    ap.add_argument('--features',choices=['six','full','both'],default='both')
    ap.add_argument('--device',choices=['auto','cpu','cuda'],default='auto')
    ap.add_argument('--epochs',type=int,default=35)
    ap.add_argument('--patience',type=int,default=6)
    ap.add_argument('--batch-size',type=int,default=512)
    ap.add_argument('--seed',type=int,default=42)
    ap.add_argument('--smoke',action='store_true')
    args=ap.parse_args()
    if min(args.epochs,args.patience,args.batch_size)<1:
        ap.error('positive epochs, patience, batch size required')
    device=torch.device(('cuda' if torch.cuda.is_available() else 'cpu') if args.device=='auto' else args.device)
    if device.type=='cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; use --device cpu')
    torch.set_num_threads(4)
    print(f'Device: {device}',flush=True)
    (train,test),hashes=load(args.data_dir,args.smoke)
    y=(train.income=='>50K').astype(int); yt=(test.income=='>50K').astype(int)
    fi,vi=train_test_split(np.arange(len(train)),test_size=.2,stratify=y,random_state=args.seed)
    out=args.out; out.mkdir(parents=True,exist_ok=True)
    rows=[]; history=[]; predictions=pd.DataFrame({'row_id':np.arange(len(test)),'target':yt})
    tags=['six','full'] if args.features=='both' else [args.features]
    meta=dict(dataset='fixture' if args.smoke else 'UCI Adult official split',n_train=len(train),
        n_validation=len(vi),n_test=len(test),seed=args.seed,threshold=.5,sha256=hashes,
        feature_sets=tags,architecture='FT-Transformer-style: numeric affine tokens, categorical embeddings, CLS, 2 pre-norm blocks, 4 heads, d=48, GELU FFN=96',
        optimizer='AdamW',lr=.001,weight_decay=.0001,batch_size=args.batch_size,
        max_epochs=args.epochs,patience=args.patience,torch=str(torch.__version__),device=str(device),
        protocol='Validation BCE early stopping; fresh refit on all official training rows for selected epochs')
    (out/'metadata.json').write_text(json.dumps(meta,indent=2),encoding='utf8')
    for tag in tags:
        features=SIX if tag=='six' else ALL
        seed_everything(args.seed)
        pre=make_pre(features); pre.fit(train.iloc[fi][features])
        x=encode(pre,train.iloc[fi][features],device); xv=encode(pre,train.iloc[vi][features],device)
        yf=labels(y.iloc[fi],device); yv=labels(y.iloc[vi],device)
        cfg=config(pre,features); net=FTTransformer(**cfg).to(device)
        opt=torch.optim.AdamW(net.parameters(),lr=.001,weight_decay=.0001)
        best=float('inf'); best_epoch=0
        for epoch in range(1,args.epochs+1):
            loss=train_epoch(net,x,yf,opt,args.batch_size)
            net.eval()
            with torch.inference_mode():
                vl=sum(float(nn.functional.binary_cross_entropy_with_logits(net(a),b,reduction='sum').cpu())
                    for a,b in zip(xv.split(1024),yv.split(1024)))/len(xv)
            vs=scores(y.iloc[vi],predict(net,xv))
            history.append(dict(features=tag,epoch=epoch,train_loss=loss,val_loss=vl,val_f1=vs['f1'],val_auc=vs['roc_auc']))
            if epoch==1 or epoch%5==0:
                print(f'{tag} epoch {epoch}: train={loss:.4f} val={vl:.4f} F1={vs["f1"]:.4f}',flush=True)
            if vl<best-1e-5:
                best=vl; best_epoch=epoch; best_scores=vs
            if epoch-best_epoch>=args.patience: break
        rows.append(dict(model='ft',features=tag,split='validation',n=len(vi),epochs=best_epoch,**best_scores))
        seed_everything(args.seed)
        pre=make_pre(features); pre.fit(train[features]); cfg=config(pre,features)
        x=encode(pre,train[features],device); yf=labels(y,device)
        net=FTTransformer(**cfg).to(device); opt=torch.optim.AdamW(net.parameters(),lr=.001,weight_decay=.0001)
        if device.type=='cuda': torch.cuda.synchronize()
        start=time.perf_counter()
        for _ in range(best_epoch): train_epoch(net,x,yf,opt,args.batch_size)
        if device.type=='cuda': torch.cuda.synchronize()
        seconds=time.perf_counter()-start
        pt=predict(net,encode(pre,test[features],device)); ts=scores(yt,pt)
        rows.append(dict(model='ft',features=tag,split='test',n=len(test),epochs=best_epoch,train_seconds=seconds,**ts))
        predictions[f'{tag}_ft']=pt
        joblib.dump(pre,out/f'{tag}_preprocessor.joblib',compress=3)
        torch.save(dict(features=features,config=cfg,state_dict={k:v.cpu() for k,v in net.state_dict().items()}),out/f'{tag}_ft.pt')
        pd.DataFrame(rows).to_csv(out/'metrics.csv',index=False)
        pd.DataFrame(history).to_csv(out/'history.csv',index=False)
        predictions.to_csv(out/'test_predictions.csv',index=False)
        restored=FTPredictor(out,tag,device)
        np.testing.assert_allclose(restored.predict_proba(test[features].iloc[:16])[:,1],pt[:16],rtol=1e-4,atol=1e-5)
        print(f'{tag}_ft test F1={ts["f1"]:.4f} AUC={ts["roc_auc"]:.4f} accuracy={ts["accuracy"]:.4f} epochs={best_epoch} fit={seconds:.1f}s',flush=True)
    print(f'Results: {out.resolve()}')


if __name__=='__main__': main()
