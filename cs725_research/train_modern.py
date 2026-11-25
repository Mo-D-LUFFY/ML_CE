"""Modern benchmarks: ACS 2024 (selected US states), India PLFS 2025.
Separate schemas/targets; group-disjoint train/validation/test; no V2 overwrite.
"""
import argparse
import gc
import hashlib
import json
import time
import urllib.request
import zipfile
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GroupShuffleSplit
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from train_classical import ROOT, scores
from train_mlp import build_mlp, seed_everything, encode, labels, train_epoch
from train_transformer import FTTransformer

CACHE=ROOT/'external_data'
ACS_BASE='https://www2.census.gov/programs-surveys/acs/data/pums/2024/1-Year/'
PLFS_URL='https://raw.githubusercontent.com/Vonter/india-plf-survey/main/data/2025.parquet'
MODELS=['lr','rf','xgb','mlp','ft']
def download(url,path):
    path.parent.mkdir(parents=True,exist_ok=True)
    if not path.exists():
        print('Downloading',url,flush=True)
        tmp=path.with_suffix(path.suffix+'.part')
        with urllib.request.urlopen(url,timeout=180) as r, tmp.open('wb') as f:
            import shutil
            shutil.copyfileobj(r,f)
        tmp.replace(path)
    return path
def cat(x):
    return x.astype('string').fillna('MISSING').astype(str)
def numeric(x):
    return pd.to_numeric(x,errors='coerce')
def file_hash(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()
def acs(states):
    cols=['SERIALNO','AGEP','COW','SCHL','MAR','OCCP','POBP','RELSHIPP','WKHP','SEX','RAC1P','PINCP','ADJINC','PWGTP']
    frames=[]; sources=[]
    for state in states:
        url=ACS_BASE+'csv_p'+state.lower()+'.zip'
        path=download(url,CACHE/('acs2024_'+state.lower()+'.zip'))
        with zipfile.ZipFile(path) as z:
            names=[n for n in z.namelist() if n.lower().endswith('.csv') and Path(n).name.lower().startswith('psam_p')]
            if not names: raise ValueError('No person CSV in '+str(path))
            for name in names:
                with z.open(name) as f:
                    d=pd.read_csv(f,usecols=cols,dtype={'SERIALNO':str,'OCCP':str,'POBP':str},low_memory=False)
                income=numeric(d.PINCP)*numeric(d.ADJINC)/1e6
                keep=(numeric(d.AGEP)>16)&(numeric(d.WKHP)>=1)&(income>100)&(numeric(d.PWGTP)>=1)
                d=d.loc[keep].copy()
                x=pd.DataFrame({'age':numeric(d.AGEP),'education_level':numeric(d.SCHL),
                    'hours':numeric(d.WKHP),'workclass':cat(d.COW),
                    'occupation':cat(d.OCCP),'marital':cat(d.MAR),'birthplace':cat(d.POBP),
                    'relationship':cat(d.RELSHIPP),'sex':cat(d.SEX),'race':cat(d.RAC1P),
                    'state':state.upper(),'target':(income.loc[keep]>50000).astype(int),
                    'group':state.upper()+':'+cat(d.SERIALNO)})
                frames.append(x)
        sources.append(dict(url=url,sha256=file_hash(path)))
    return pd.concat(frames,ignore_index=True), ['age','education_level','hours'],dict(
        dataset='ACS PUMS 2024',states=states,sources=sources,
        target='ADJINC-adjusted annual personal income > USD 50000',threshold=50000,
        filter='AGEP>16, WKHP>=1, adjusted PINCP>100, PWGTP>=1; adapted ACSIncome task',
        groups='state + household SERIALNO',weights='unweighted prediction benchmark, not population estimates')
def plfs(path):
    if path is None: path=download(PLFS_URL,CACHE/'plfs2025.parquet')
    cols=['Person ID','Survey Year','State/UT','First Stage Unit (FSU)','Age','Sex','Marital Status',
        'General Education Level','Years of Formal Education','CWS Status','CWS Occupation','CWS Industry',
        'CWS Earnings (Salaried)','CWS Total Weekly Hours','Sector','Relationship to Head']
    d=pd.read_parquet(path,columns=cols)
    if not d['Survey Year'].eq(2025).all(): raise ValueError('Expected PLFS 2025')
    if d['Person ID'].duplicated().any(): raise ValueError('Duplicate source person IDs')
    salary=numeric(d['CWS Earnings (Salaried)'])
    status=cat(d['CWS Status'])
    salaried=status.str.contains('regular|salaried',case=False,regex=True)
    if not salaried.any(): raise ValueError('Unrecognized salaried status; inspect: '+str(status.value_counts().head(30).to_dict()))
    keep=(numeric(d.Age)>=18)&(salary>0)&salaried&(numeric(d['CWS Total Weekly Hours'])>=1)
    d=d.loc[keep].copy()
    x=pd.DataFrame({'age':numeric(d.Age),'education_level':numeric(d['Years of Formal Education']),
        'education':cat(d['General Education Level']),'hours':numeric(d['CWS Total Weekly Hours']),
        'workclass':cat(d['CWS Industry']).str.slice(0,2), 'occupation':cat(d['CWS Occupation']),
        'marital':cat(d['Marital Status']),'sex':cat(d.Sex),'state':cat(d['State/UT']),
        'sector':cat(d.Sector),'relationship':cat(d['Relationship to Head']),
        'target':(salary.loc[keep]>30000).astype(int),
        'group':cat(d['State/UT'])+':'+cat(d['First Stage Unit (FSU)'])})
    return x.reset_index(drop=True),['age','education_level','hours'],dict(dataset='India PLFS 2025',
        source=PLFS_URL,sha256=file_hash(path),source_rows=int(len(keep)),
        attribution='MoSPI PLFS 2025; harmonized by Vonter/india-plf-survey, ODbL 1.0',
        target='reported monthly salaried earnings > INR 30000',threshold=30000,
        filter='age>=18, CWS regular/salaried status, positive salaried earnings, weekly hours>=1',
        groups='state + FSU (survey cluster)',weights='unweighted prediction benchmark; not population estimates',
        workclass='CWS industry two-character prefix, not Adult employer category')
def preprocessor(features,num,transformer=False):
    cats=[f for f in features if f not in num]
    enc=OrdinalEncoder(handle_unknown='use_encoded_value',unknown_value=-1) if transformer else OneHotEncoder(handle_unknown='ignore')
    return ColumnTransformer([('num',make_pipeline(SimpleImputer(strategy='median'),StandardScaler()),num),
        ('cat',make_pipeline(SimpleImputer(strategy='constant',fill_value='MISSING'),enc),cats)])
def classical(name,features,num,seed):
    if name=='lr': m=LogisticRegression(max_iter=1500,random_state=seed)
    elif name=='rf': m=RandomForestClassifier(n_estimators=120,min_samples_leaf=5,n_jobs=4,random_state=seed)
    else: m=XGBClassifier(n_estimators=300,max_depth=5,learning_rate=.08,subsample=.85,
        colsample_bytree=.85,reg_lambda=1,tree_method='hist',eval_metric='logloss',random_state=seed,n_jobs=4)
    return make_pipeline(preprocessor(features,num),m)
def neural_config(name,pre,num,dim):
    if name=='mlp': return {'input_dim':dim}
    enc=pre.named_transformers_['cat'].steps[-1][1]
    return dict(n_num=len(num),cards=[len(v)+1 for v in enc.categories_],dim=48,heads=4,layers=2,dropout=.1)
def net_build(name,cfg):
    return build_mlp(**cfg) if name=='mlp' else FTTransformer(**cfg)
@torch.inference_mode()
def net_predict(net,x):
    net.eval()
    return torch.cat([net(b).sigmoid().flatten().cpu() for b in x.split(1024)]).numpy()
def neural(name,df,y,fit,val,train,test,features,num,args,device,out):
    seed_everything(args.seed)
    pre=preprocessor(features,num,name=='ft'); pre.fit(df.iloc[fit][features])
    x=encode(pre,df.iloc[fit][features],device); xv=encode(pre,df.iloc[val][features],device)
    yf=labels(y.iloc[fit],device); yv=labels(y.iloc[val],device)
    cfg=neural_config(name,pre,num,x.shape[1]); net=net_build(name,cfg).to(device)
    opt=torch.optim.AdamW(net.parameters(),lr=.001,weight_decay=.0001)
    best=float('inf'); chosen=1; hist=[]
    for epoch in range(1,args.epochs+1):
        loss=train_epoch(net,x,yf,opt,512); net.eval()
        with torch.inference_mode():
            vl=sum(float(nn.functional.binary_cross_entropy_with_logits(net(a),b,reduction='sum').cpu())
                for a,b in zip(xv.split(1024),yv.split(1024)))/len(xv)
        hist.append(dict(epoch=epoch,train_loss=loss,val_loss=vl))
        if vl<best-1e-5: best=vl; chosen=epoch
        if epoch==1 or epoch%5==0: print(name,epoch,round(loss,4),round(vl,4),flush=True)
        if epoch-chosen>=5: break
    pd.DataFrame(hist).to_csv(out/(name+'_history.csv'),index=False)
    del x,xv,yf,yv,net,opt,pre
    if device.type=='cuda': torch.cuda.empty_cache()
    seed_everything(args.seed)
    pre=preprocessor(features,num,name=='ft'); pre.fit(df.iloc[train][features])
    x=encode(pre,df.iloc[train][features],device); yf=labels(y.iloc[train],device)
    cfg=neural_config(name,pre,num,x.shape[1]); net=net_build(name,cfg).to(device)
    opt=torch.optim.AdamW(net.parameters(),lr=.001,weight_decay=.0001)
    if device.type=='cuda': torch.cuda.synchronize()
    t=time.perf_counter()
    for _ in range(chosen): train_epoch(net,x,yf,opt,512)
    if device.type=='cuda': torch.cuda.synchronize()
    elapsed=time.perf_counter()-t
    p=net_predict(net,encode(pre,df.iloc[test][features],device))
    joblib.dump(pre,out/(name+'_pre.joblib'),compress=3)
    torch.save(dict(config=cfg,state_dict={k:v.cpu() for k,v in net.state_dict().items()}),out/(name+'.pt'))
    del x,yf,opt,net,pre
    if device.type=='cuda': torch.cuda.empty_cache()
    return p,elapsed,chosen
class Neural:
    def __init__(self,name,out,features,device):
        self.features=features; self.device=device
        self.pre=joblib.load(out/(name+'_pre.joblib'))
        ck=torch.load(out/(name+'.pt'),map_location=device,weights_only=True)
        self.net=net_build(name,ck['config']).to(device); self.net.load_state_dict(ck['state_dict'])
    def predict_proba(self,x):
        p=net_predict(self.net,encode(self.pre,x[self.features],self.device))
        return np.column_stack((1-p,p))
def modern_cf(df,train,test,features,num,models,p0,args,out):
    """Matched attempts, shared ranking; domain-specific observed group edits."""
    rng=np.random.default_rng(args.seed); tr=df.iloc[train][features].reset_index(drop=True)
    te=df.iloc[test][features].reset_index(drop=True)
    eligible=np.flatnonzero((p0[:,:5]<.5).all(axis=1))
    ids=rng.choice(eligible,min(len(eligible),args.queries),replace=False)
    ed=['education_level']+(['education'] if 'education' in features else [])
    job=['workclass','occupation']
    groups=[[f] for f in features if f not in ed+job]+[ed,job]
    scale=(tr[num].quantile(.95)-tr[num].quantile(.05)).clip(lower=1)
    cats=[f for f in features if f not in num]
    education_pairs=set(map(tuple,tr[ed].dropna().to_numpy()))
    job_pairs=set(map(tuple,tr[job].to_numpy()))
    # Never change unknown category or missing education to a fabricated value.
    known_edu=tr[ed].dropna().drop_duplicates()
    if 'education' in ed: known_edu=known_edu[known_edu.education!='MISSING']
    known_jobs=tr[job].drop_duplicates()
    known_jobs=known_jobs[(known_jobs!='MISSING').all(axis=1)]
    rows=[]; case=None
    for i,q in enumerate(ids):
        x=te.iloc[q]; h=float(x.hours); e=float(x.education_level)
        hvals=list(range(int(h)+1,min(int(h)+15,65)+1))
        ep=known_edu[(known_edu.education_level>e)&(known_edu.education_level<=e+2)]
        for method in ['unconstrained','constrained','hours_only']:
            n=len(hvals) if method=='hours_only' else args.budget
            if n==0:
                a=pd.DataFrame(columns=features)
            else:
                a=pd.DataFrame([x.to_dict()]*n,columns=features)
                if method=='hours_only': a['hours']=hvals
                else:
                    donor=tr.iloc[rng.integers(len(tr),size=n)].reset_index(drop=True)
                    gs=groups if method=='unconstrained' else [['hours'],ed,job]
                    for k in range(n):
                        for gidx in rng.choice(len(gs),size=int(rng.integers(1,min(3,len(gs))+1)),replace=False):
                            g=gs[gidx]
                            if method=='unconstrained': a.loc[k,g]=donor.loc[k,g].to_numpy()
                            elif g==['hours'] and hvals: a.at[k,'hours']=rng.choice(hvals)
                            elif g==ed and len(ep): a.loc[k,g]=ep.iloc[rng.integers(len(ep))].to_numpy()
                            elif g==job and len(known_jobs): a.loc[k,g]=known_jobs.iloc[rng.integers(len(known_jobs))].to_numpy()
            if len(a):
                for f in num: a[f]=pd.to_numeric(a[f],errors='coerce')
                p=np.column_stack([m.predict_proba(a)[:,1] for m in models]); p=np.column_stack([p,p.mean(axis=1)])
                changed=a.ne(x)&~(a.isna()&x.isna())
                dh=a.hours-h; de=a.education_level-e; ec=changed[ed].any(axis=1); jc=changed[job].any(axis=1)
                fixed=[f for f in features if f not in ed+job+['hours']]
                h_ok=(~changed.hours)|((dh>0)&(dh<=15)&(a.hours<=65))
                e_ok=(~ec)|((de>0)&(de<=2)&pd.Series([v in education_pairs for v in map(tuple,a[ed].to_numpy())]) & ((a.education!='MISSING') if 'education' in ed else True))
                j_ok=(~jc)|pd.Series([v in job_pairs and 'MISSING' not in v for v in map(tuple,a[job].to_numpy())])
                feasible=(~changed[fixed].any(axis=1)&h_ok&e_ok&j_ok).to_numpy()
                cost=(dh.fillna(0).clip(lower=0)+10*de.fillna(0).clip(lower=0)+20*jc).to_numpy()
                dist=(((a[num]-x[num].astype(float)).abs()/scale).fillna(1).sum(axis=1)+changed[cats].sum(axis=1)).to_numpy()/len(features)
                caution=(ec|jc|(dh>10)).to_numpy()
            else:
                p=np.empty((0,6)); feasible=np.empty(0,bool); cost=dist=np.empty(0); caution=np.empty(0,bool)
            for j,name in enumerate(MODELS+['ensemble','joint']):
                valid=(p[:,:5]>=.5).all(axis=1) if name=='joint' else p[:,j]>=.5
                ix=np.flatnonzero(valid & (feasible if method!='unconstrained' or name=='joint' else True))
                k=int(ix[np.lexsort((dist[ix],cost[ix]))[0]]) if len(ix) else None
                rows.append(dict(query_id=int(test[q]),method=method,target=name,found=int(k is not None),
                    feasible=int(k is not None and feasible[k]),robust=int(k is not None and (p[k,:5]>=.5).all()),
                    cost=float(cost[k]) if k is not None and feasible[k] else np.nan,
                    caution=int(k is not None and feasible[k] and caution[k]),attempts=n))
                if case is None and name=='joint' and method=='constrained' and k is not None:
                    case=dict(query_id=int(test[q]),changes={f:[str(x[f]),str(a.iloc[k][f])] for f in features if str(x[f])!=str(a.iloc[k][f])},
                        before=p0[q].tolist(),after=p[k].tolist(),cost=float(cost[k]),caution=bool(caution[k]))
        if (i+1)%25==0: print('Modern CF',i+1,'/',len(ids),flush=True)
    if rows:
        r=pd.DataFrame(rows); r.to_csv(out/'cf_per_query.csv',index=False)
        r.groupby(['method','target']).agg(queries=('found','size'),coverage=('found','mean'),
            feasible_coverage=('feasible','mean'),robust_coverage=('robust','mean'),median_cost=('cost','median')).reset_index().to_csv(out/'cf_summary.csv',index=False)
    (out/'cf_metadata.json').write_text(json.dumps(dict(eligible_common_negative=len(eligible),queries=len(ids),budget=args.budget,
        policy='Freeze remaining features; hours +1..15 up to 65; observed education pair increase <=2; observed nonmissing employment pair; conditional edits unverified',
        cost='hours + 10*education level/years + 20*employment pair change; illustrative, not comparable effort across countries',
        method='Equal proposal attempts for random conditions; same cost/distance ranking, joint always policy-constrained; exact hours reference; no expanded exact search'),indent=2))
    (out/'cf_case.json').write_text(json.dumps(case,indent=2))
def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('dataset',choices=['acs','plfs'])
    ap.add_argument('--states',nargs='+',default=['CA','TX'])
    ap.add_argument('--plfs-file',type=Path)
    ap.add_argument('--epochs',type=int,default=20)
    ap.add_argument('--queries',type=int,default=200)
    ap.add_argument('--budget',type=int,default=256)
    ap.add_argument('--seed',type=int,default=42)
    ap.add_argument('--device',choices=['auto','cpu','cuda'],default='auto')
    ap.add_argument('--skip-cf',action='store_true')
    args=ap.parse_args()
    if min(args.epochs,args.queries,args.budget)<1: ap.error('positive epochs, queries, budget required')
    torch.set_num_threads(4); device=torch.device(('cuda' if torch.cuda.is_available() else 'cpu') if args.device=='auto' else args.device)
    out=ROOT/'results'/'modern'/args.dataset; out.mkdir(parents=True,exist_ok=True)
    df,num,meta=acs(args.states) if args.dataset=='acs' else plfs(args.plfs_file)
    # Keep the original eligible rows; no synthetic expansion or cap.
    df=df[df[num].notna().all(axis=1)].reset_index(drop=True)
    features=[f for f in df if f not in ['target','group']]
    y=df.target.astype(int); groups=df.group
    if len(df)<1000 or y.nunique()!=2: raise ValueError('Insufficient eligible data/classes')
    train,test=next(GroupShuffleSplit(n_splits=1,test_size=.2,random_state=args.seed).split(df,y,groups))
    f,v=next(GroupShuffleSplit(n_splits=1,test_size=.2,random_state=args.seed).split(df.iloc[train],y.iloc[train],groups.iloc[train]))
    fit,val=train[f],train[v]
    assert not (set(groups.iloc[train])&set(groups.iloc[test]))
    assert not (set(groups.iloc[fit])&set(groups.iloc[val]))
    if any(y.iloc[k].nunique()!=2 for k in [fit,val,test]): raise ValueError('A split lacks one class')
    meta.update(n_eligible=len(df),n_train=len(train),n_validation=len(val),n_test=len(test),
        test_positive=int(y.iloc[test].sum()),features=features,numeric=num,seed=args.seed,device=str(device),
        ensemble='unweighted mean of five positive scores',split='group-disjoint 80/20 test; inner group-disjoint 80/20 validation; final refit',
        max_epochs=args.epochs,policy_warning='Research comparison only; no causal interpretation or actual verified job/education opportunity')
    (out/'metadata.json').write_text(json.dumps(meta,indent=2))
    print(meta,flush=True)
    np.savez_compressed(out/'split_indices.npz',train=train,test=test,fit=fit,val=val)
    rows=[]; ps=[]; fitted=[]
    for name in MODELS:
        print('Training',name,flush=True)
        if name in MODELS[:3]:
            m=classical(name,features,num,args.seed); t=time.perf_counter()
            m.fit(df.iloc[train][features],y.iloc[train]); elapsed=time.perf_counter()-t
            p=m.predict_proba(df.iloc[test][features])[:,1]; chosen=None
            joblib.dump(m,out/(name+'.joblib'),compress=3); fitted.append(m)
        else:
            p,elapsed,chosen=neural(name,df,y,fit,val,train,test,features,num,args,device,out)
            fitted.append(Neural(name,out,features,device))
        ps.append(p); rows.append(dict(model=name,epochs=chosen,train_seconds=elapsed,**scores(y.iloc[test],p)))
        pd.DataFrame(rows).to_csv(out/'metrics.csv',index=False)
        print(rows[-1],flush=True); gc.collect()
    p=np.column_stack(ps); p=np.column_stack([p,p.mean(axis=1)])
    rows.append(dict(model='ensemble',**scores(y.iloc[test],p[:,-1])))
    pd.DataFrame(rows).to_csv(out/'metrics.csv',index=False)
    pd.DataFrame(dict(row_id=test,target=y.iloc[test].to_numpy(),**{n:p[:,j] for j,n in enumerate(MODELS+['ensemble'])})).to_csv(out/'test_predictions.csv',index=False)
    if not args.skip_cf: modern_cf(df,train,test,features,num,fitted,p,args,out)
    print('Results:',out,flush=True)
if __name__=='__main__': main()
