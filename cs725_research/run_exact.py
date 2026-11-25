"""Exhaustive reference over the finite expanded Adult action space.
Uses the same query IDs and policy as the recorded random-search benchmark.
"""
import argparse
import itertools
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from run_counterfactuals import (ROOT, ALL, NUMI, NAMES, H, E, ED, JOB, MISSING,
    load, clean, models, probabilities, assess, mixed_distance, rank)


def enumerate_candidates(x, edu_pairs, job_pairs):
    hours=[int(x[H])]+list(range(int(x[H])+1,min(65,int(x[H])+15)+1))
    edu=[(x[ED],x[E])]+sorted(
        [(a,b) for a,b in edu_pairs if 1<=float(b)-float(x[E])<=2],
        key=lambda p:(float(p[1]),str(p[0])))
    jobs=[tuple(x[JOB])]+sorted([p for p in job_pairs
        if MISSING not in p and p!=tuple(x[JOB])],key=lambda p:tuple(map(str,p)))
    values=list(itertools.product(hours,edu,jobs))
    a=np.tile(x,(len(values),1))
    for i,(h,(ed,e),job) in enumerate(values):
        a[i,H]=h; a[i,ED]=ed; a[i,E]=e; a[i,JOB]=job
    a=a[(a!=x).any(axis=1)]
    return a


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--results',type=Path,default=ROOT/'results')
    ap.add_argument('--data-dir',type=Path,default=ROOT.parent/'cs725_starter'/'data')
    ap.add_argument('--queries',type=int,default=1000)
    ap.add_argument('--device',choices=['auto','cpu','cuda'],default='auto')
    ap.add_argument('--out',type=Path,default=None)
    args=ap.parse_args()
    if args.queries<1: ap.error('queries must be positive')
    root=args.results; out=args.out or root/'exact'; out.mkdir(parents=True,exist_ok=True)
    old=pd.read_csv(root/'counterfactual'/'per_query.csv')
    meta=json.loads((root/'counterfactual'/'metadata.json').read_text(encoding='utf8'))
    (train,test),hashes=load(args.data_dir)
    if hashes!=meta['sha256'] or meta['threshold']!=.5:
        raise ValueError('Mismatch with existing counterfactual experiment')
    ids=old.query_id.drop_duplicates().to_numpy()[:args.queries]
    if len(ids)==0: raise ValueError('No query IDs')
    device=('cuda' if torch.cuda.is_available() else 'cpu') if args.device=='auto' else args.device
    torch.set_num_threads(4)
    ms=models(root,device)
    tr,te=clean(train),clean(test)
    p0=probabilities(ms,te[ids])
    if not (p0<.5).all(): raise ValueError('Original cohort no longer common negative')
    edu_pairs=set(zip(tr[:,ED],tr[:,E])); job_pairs={tuple(v) for v in tr[:,JOB]}
    scale=np.maximum(np.quantile(tr[:,NUMI].astype(float),.95,axis=0)-
                     np.quantile(tr[:,NUMI].astype(float),.05,axis=0),1)
    baseline=old[old.method.isin(['joint_robust','constrained_random'])].set_index(['query_id','source_model'])
    metadata=dict(dataset=meta['dataset'],sha256=hashes,n_queries=len(ids),seed=meta['seed'],
        threshold=.5,device=device,policy=meta['expanded_policy'],cost=meta['cost'],
        interpretation='Exhaustive only within the specified discrete domain; conditional actions are not verified, and no causal claim is made.',
        comparison='Same query IDs and cost/distance ranking as the random constrained search. Exact search is a reference with a larger variable budget.')
    (out/'metadata.json').write_text(json.dumps(metadata,indent=2),encoding='utf8')
    rows=[]
    def save():
        d=pd.DataFrame(rows); d.to_csv(out/'per_query.csv',index=False)
        summary=d.groupby('source_model',sort=False).agg(queries=('found','size'),
            random_coverage=('random_found','mean'),exact_coverage=('found','mean'),
            extra_coverage=('extra','mean'),median_exact_cost=('cost','median'),
            mean_candidates=('candidates','mean'),median_seconds=('seconds','median'),
            hours_only_coverage=('hours_only_found','mean')).reset_index()
        summary.to_csv(out/'summary.csv',index=False)
    for i,q in enumerate(ids):
        start=time.perf_counter(); x=te[q]
        a=enumerate_candidates(x,edu_pairs,job_pairs)
        if len(a):
            feasible,caution,cost,edits,reason=assess(a,x,edu_pairs,job_pairs)
            if not feasible.all(): raise AssertionError('Enumeration violated policy')
            p=probabilities(ms,a); dist=mixed_distance(a,x,scale)
            hours_only=(a[:,H]!=x[H]) & (edits==1)
        else:
            p=np.empty((0,len(NAMES))); cost=dist=np.empty(0); hours_only=np.empty(0,bool)
        for j,n in enumerate(NAMES+['joint']):
            valid=(p>=.5).all(axis=1) if n=='joint' else p[:,j]>=.5
            k=rank(np.flatnonzero(valid),dist,cost,True)
            b=baseline.loc[(q,n)]; old_found=int(b.found)
            found=int(k is not None)
            new_cost=float(cost[k]) if found else np.nan
            # Exact coverage should dominate the sampled search, up to floating-point threshold effects.
            if old_found and (not found or new_cost>float(b.cost)+1e-8):
                raise AssertionError(f'Exact search failed to dominate random for query {q}, {n}')
            rows.append(dict(query_id=int(q),source_model=n,found=found,
                random_found=old_found,extra=found-old_found,cost=new_cost,
                random_cost=float(b.cost) if old_found else np.nan,
                hours_only_found=int((valid&hours_only).any()),
                candidates=len(a),seconds=time.perf_counter()-start))
        if (i+1)%25==0 or i==0:
            print(f'Exact queries {i+1}/{len(ids)}',flush=True); save()
    save(); print(pd.read_csv(out/'summary.csv').to_string(index=False))
    print(f'Results: {out.resolve()}')


if __name__=='__main__': main()
