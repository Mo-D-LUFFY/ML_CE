"""Milestone 4: matched-budget counterfactual and cross-model robustness benchmark."""
import argparse
import json
import time
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import torch
from train_classical import ROOT, ALL, NUM, load
from train_mlp import MLPPredictor
from train_transformer import FTPredictor

NAMES = ['lr','rf','xgb','mlp','ft']
IDX = {f:i for i,f in enumerate(ALL)}
H,E,ED = [IDX[f] for f in ['hours-per-week','education-num','education']]
JOB = [IDX[f] for f in ['workclass','occupation']]
ACTION = {H,E,ED,*JOB}
GROUPS = [[IDX[f] for f in g] for g in [
    ['age'],['workclass','occupation'],['fnlwgt'],['education','education-num'],
    ['marital-status','relationship'],['race'],['sex'],['capital-gain'],
    ['capital-loss'],['hours-per-week'],['native-country']]]
NUMI = [IDX[f] for f in NUM]
CATI = [i for i in range(len(ALL)) if i not in NUMI]
MISSING = '__MISSING__'


def clean(frame):
    a=frame[ALL].astype(object).to_numpy(copy=True)
    a[pd.isna(a)]=MISSING
    return a


def frame(a):
    d=pd.DataFrame(a,columns=ALL)
    for f in NUM: d[f]=pd.to_numeric(d[f])
    for f in set(ALL)-set(NUM): d[f]=d[f].replace(MISSING,np.nan)
    return d


def models(root,device):
    result={n:joblib.load(root/'classical'/'models'/f'full_{n}.joblib') for n in NAMES[:3]}
    for n in NAMES[:3]:
        clf=result[n].steps[-1][1]
        if hasattr(clf,'n_jobs'): clf.n_jobs=4
    result['mlp']=MLPPredictor(root/'mlp','full',device)
    result['ft']=FTPredictor(root/'transformer','full',device)
    return result


def probabilities(ms,a):
    d=frame(a)
    return np.column_stack([ms[n].predict_proba(d)[:,1] for n in NAMES])


def mixed_distance(a, x, scale):
    num=np.abs(a[:,NUMI].astype(float)-x[NUMI].astype(float))/scale
    cat=(a[:,CATI]!=x[CATI]).astype(float)
    return (num.sum(axis=1)+cat.sum(axis=1))/len(ALL)


def assess(a,x,edu_pairs,job_pairs):
    """Expanded hypothetical policy; conditional changes are not verified actions."""
    changed=a!=x
    delta_h=a[:,H].astype(float)-float(x[H])
    delta_e=a[:,E].astype(float)-float(x[E])
    fixed=[i for i in range(len(ALL)) if i not in ACTION]
    frozen=changed[:,fixed].any(axis=1)
    h_bad=changed[:,H]&((delta_h<0)|(delta_h>15)|(a[:,H].astype(float)>65)|(a[:,H].astype(float)<1))
    e_changed=changed[:,[E,ED]].any(axis=1)
    e_bad=e_changed&((delta_e<1)|(delta_e>2)|np.array([(v[ED],v[E]) not in edu_pairs for v in a]))
    j_changed=changed[:,JOB].any(axis=1)
    j_bad=j_changed&np.array([tuple(v[JOB]) not in job_pairs for v in a])
    j_bad|=j_changed&np.array([MISSING in v[JOB] for v in a])
    feasible=~(frozen|h_bad|e_bad|j_bad)
    caution=(delta_h>10)|e_changed|j_changed
    cost=np.maximum(delta_h,0)+10*np.maximum(delta_e,0)+20*j_changed
    reason=np.where(frozen,'frozen_feature',np.where(h_bad,'hours_policy',np.where(e_bad,'education_policy',np.where(j_bad,'job_policy',''))))
    return feasible,caution,cost,changed.sum(axis=1),reason


def propose(x,train,budget,rng,constrained):
    """Same donor-group search mechanism in both conditions; fixed attempts."""
    a=np.tile(x,(budget,1))
    donors=train[rng.integers(len(train),size=budget)]
    groups=[[H],[E,ED],JOB] if constrained else GROUPS
    for i in range(budget):
        count=int(rng.integers(1,min(3,len(groups))+1))
        for g in rng.choice(len(groups),size=count,replace=False):
            cols=groups[g]
            a[i,cols]=donors[i,cols]
            if constrained and cols==[H]:
                upper=min(65,int(x[H])+15)
                a[i,H]=rng.integers(int(x[H])+1,upper+1) if upper>int(x[H]) else x[H]
            elif constrained and cols==[E,ED]:
                # Draw a coherent education pair from the observed training data.
                if not 1<=float(a[i,E])-float(x[E])<=2:
                    a[i,cols]=x[cols]
            elif constrained and cols==JOB and MISSING in a[i,cols]:
                a[i,cols]=x[cols]
    return a


def strict(x):
    h=int(x[H])
    vals=np.arange(h+1,min(h+15,65)+1)
    a=np.tile(x,(len(vals),1))
    a[:,H]=vals
    return a


def rank(indices,dist,cost,constrained):
    if len(indices)==0: return None
    # Use distance for the unconstrained method; same effort/distance ranking
    # for single-model and joint-model constrained searches.
    if constrained:
        return int(indices[np.lexsort((dist[indices],cost[indices]))[0]])
    return int(indices[np.argmin(dist[indices])])


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data-dir',type=Path,default=ROOT.parent/'cs725_starter'/'data')
    ap.add_argument('--results',type=Path,default=ROOT/'results')
    ap.add_argument('--out',type=Path,default=None)
    ap.add_argument('--queries',type=int,default=500)
    ap.add_argument('--budget',type=int,default=256)
    ap.add_argument('--seed',type=int,default=42)
    ap.add_argument('--device',choices=['auto','cpu','cuda'],default='auto')
    args=ap.parse_args()
    if min(args.queries,args.budget)<1: ap.error('positive queries and budget required')
    out=args.out or args.results/'counterfactual'; out.mkdir(parents=True,exist_ok=True)
    device=('cuda' if torch.cuda.is_available() else 'cpu') if args.device=='auto' else args.device
    if device=='cuda' and not torch.cuda.is_available(): raise RuntimeError('CUDA unavailable')
    torch.set_num_threads(4)
    (train,test),hashes=load(args.data_dir)
    for folder in ['classical','mlp','transformer']:
        meta=json.loads((args.results/folder/'metadata.json').read_text(encoding='utf8'))
        if meta['sha256']!=hashes or meta['dataset']!='UCI Adult official split':
            raise ValueError(f'Dataset mismatch in {folder}')
    tr,te=clean(train),clean(test)
    ms=models(args.results,device)
    ptest=probabilities(ms,te)
    # Verify restored checkpoints against the saved test scores.
    for folder,ns in [('classical',NAMES[:3]),('mlp',['mlp']),('transformer',['ft'])]:
        saved=pd.read_csv(args.results/folder/'test_predictions.csv')
        if not np.array_equal(saved.row_id.to_numpy(),np.arange(len(te))): raise ValueError('Test row order mismatch')
        for n in ns:
            np.testing.assert_allclose(ptest[:,NAMES.index(n)],saved[f'full_{n}'],atol=2e-4,rtol=2e-4)
    eligible=np.flatnonzero((ptest<.5).all(axis=1))
    if not len(eligible): raise ValueError('No common negative cohort')
    rng=np.random.default_rng(args.seed)
    ids=rng.choice(eligible,min(args.queries,len(eligible)),replace=False)
    ptr=probabilities(ms,tr)
    positive=[np.flatnonzero(ptr[:,j]>=.5) for j in range(len(NAMES))]
    scale=np.maximum(np.quantile(tr[:,NUMI].astype(float),.95,axis=0)-np.quantile(tr[:,NUMI].astype(float),.05,axis=0),1)
    edu_pairs=set(zip(tr[:,ED],tr[:,E]))
    job_pairs={tuple(v) for v in tr[:,JOB]}
    metadata=dict(dataset='UCI Adult official split',sha256=hashes,n_train=len(tr),n_test=len(te),
        eligible_common_negative=len(eligible),n_queries=len(ids),seed=args.seed,budget=args.budget,
        threshold=.5,models=NAMES,device=device,
        methods=['nearest_positive','unconstrained_random','constrained_random','strict_hours_exact','joint_robust'],
        expanded_policy='Freeze all except hours (increase <=15, max 65), education (coherent +1 or +2 levels), workclass/occupation (observed nonmissing pair). Education and job changes conditional and unverified.',
        cost='Illustrative units: +1 per extra weekly hour, +10 per education level, +20 for job pair change.',
        comparison='Random search uses equal proposal attempts, duplicates allowed and reported. Strict hours and nearest-positive are separate exact/reference searches, not matched-budget methods. Joint method reuses constrained candidates, requires all five models positive.',
        limitation='Observational scores; no causal or actual feasibility claim. Cross-model robustness is in-sample across these five fixed predictors, not robustness to unseen retraining.')
    (out/'metadata.json').write_text(json.dumps(metadata,indent=2),encoding='utf8')
    rows=[]; cases=[]
    def record(q,method,source,x,a,p,k,dist,assessment,elapsed,attempts,unique):
        feasible,caution,cost,edits,reason=assessment
        r=dict(query_id=int(q),method=method,source_model=source,attempts=attempts,
               unique_candidates=unique,search_seconds=elapsed,found=int(k is not None),
               feasible=0,robust=0,robust_feasible=0,caution=0,cost=np.nan,edits=np.nan,distance=np.nan,reason='no_candidate')
        for name in NAMES: r['flip_'+name]=0
        if k is not None:
            flips=p[k]>=.5
            r.update(feasible=int(feasible[k]),robust=int(flips.all()),
                     robust_feasible=int(flips.all() and feasible[k]),caution=int(caution[k] and feasible[k]),
                     cost=float(cost[k]) if feasible[k] else np.nan,edits=int(edits[k]),
                     distance=float(dist[k]),reason=str(reason[k]))
            for j,name in enumerate(NAMES): r['flip_'+name]=int(flips[j])
            changes={f:{'before':str(x[i]),'after':str(a[k,i])} for i,f in enumerate(ALL) if x[i]!=a[k,i]}
            cases.append(dict(query_id=int(q),method=method,source_model=source,changes=changes,
                              original_scores=dict(zip(NAMES,ptest[q].astype(float).tolist())),
                              candidate_scores=dict(zip(NAMES,p[k].astype(float).tolist())),
                              feasible=bool(feasible[k]),caution=bool(caution[k]),reason=str(reason[k])))
        rows.append(r)
    def save():
        d=pd.DataFrame(rows); d.to_csv(out/'per_query.csv',index=False)
        summary=d.groupby(['method','source_model'],sort=False).agg(
            queries=('found','size'),flip_coverage=('found','mean'),
            feasible_coverage=('feasible','mean'),robust_coverage=('robust','mean'),
            robust_feasible_coverage=('robust_feasible','mean'),caution_coverage=('caution','mean'),
            median_feasible_cost=('cost','median'),median_edits=('edits','median'),
            median_distance=('distance','median'),mean_attempts=('attempts','mean'),
            mean_unique=('unique_candidates','mean'),median_seconds=('search_seconds','median')).reset_index()
        summary.to_csv(out/'summary.csv',index=False)
        transfer=d[d.source_model.isin(NAMES)].groupby(['method','source_model'])[['flip_'+n for n in NAMES]].mean()
        transfer.to_csv(out/'transfer.csv')
        (out/'cases.json').write_text(json.dumps(cases,indent=2),encoding='utf8')
    for i,q in enumerate(ids):
        x=te[q]
        # Reference: closest observed row predicted positive by each model.
        t=time.perf_counter()
        dd=mixed_distance(tr,x,scale)
        for j,n in enumerate(NAMES):
            ix=positive[j]
            if not len(ix): continue
            k0=ix[np.argmin(dd[ix])]; a=tr[[k0]]; p=ptr[[k0]]
            record(q,'nearest_positive',n,x,a,p,0,dd[[k0]],assess(a,x,edu_pairs,job_pairs),time.perf_counter()-t,len(tr),len(tr))
        # Random searches and exact strict-hours search; score each pool once for all models.
        for method in ['unconstrained_random','constrained_random','strict_hours_exact']:
            t=time.perf_counter()
            constrained=method!='unconstrained_random'
            a=strict(x) if method=='strict_hours_exact' else propose(x,tr,args.budget,rng,constrained)
            attempts=len(a)
            unique=len({tuple(v) for v in a})
            if len(a):
                p=probabilities(ms,a); dist=mixed_distance(a,x,scale)
                assessment=assess(a,x,edu_pairs,job_pairs)
            else:
                p=np.empty((0,len(NAMES))); dist=np.empty(0)
                assessment=(np.empty(0,bool),np.empty(0,bool),np.empty(0),np.empty(0,int),np.empty(0,object))
            feasible,_,cost,_,_=assessment
            allowed=feasible if constrained else np.ones(len(a),bool)
            for j,n in enumerate(NAMES):
                k=rank(np.flatnonzero(allowed&(p[:,j]>=.5)),dist,cost,constrained)
                record(q,method,n,x,a,p,k,dist,assessment,time.perf_counter()-t,attempts,unique)
            if method=='constrained_random':
                k=rank(np.flatnonzero(feasible&(p>=.5).all(axis=1)),dist,cost,True)
                record(q,'joint_robust','joint',x,a,p,k,dist,assessment,time.perf_counter()-t,attempts,unique)
        if (i+1)%25==0 or i==0:
            print(f'Queries {i+1}/{len(ids)}',flush=True)
            save()
    save()
    print(pd.read_csv(out/'summary.csv').to_string(index=False))
    print(f'Results: {out.resolve()}')


if __name__=='__main__': main()
