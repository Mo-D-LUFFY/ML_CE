"""Generate report figures, paired bootstrap summaries and offline dashboard. No training."""
import argparse
import html
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

MODELS = ['lr','rf','xgb','mlp','ft']
LABEL = dict(zip(MODELS+['joint'], ['LR','RF','XGBoost','MLP','FT','Joint']))
METHODS = ['nearest_positive','unconstrained_random','constrained_random','strict_hours_exact']
SHORT = ['Nearest positive','Unconstrained','Constrained','Hours only']


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--results',type=Path,default=Path(__file__).resolve().parent/'results')
    ap.add_argument('--out',type=Path,default=None)
    args=ap.parse_args(); root=args.results; out=args.out or root/'report'; out.mkdir(parents=True,exist_ok=True)
    folders=['classical','mlp','transformer','counterfactual','exact']
    metas=[json.loads((root/f/'metadata.json').read_text(encoding='utf8')) for f in folders]
    assert all(m['sha256']==metas[0]['sha256'] and m['dataset']==metas[0]['dataset'] for m in metas)
    cl=pd.concat([pd.read_csv(root/f/'metrics.csv') for f in folders[:3]],ignore_index=True)
    cl=cl[cl.split.eq('test')].copy(); assert len(cl)==10 and not cl.duplicated(['model','features']).any()
    cf=pd.read_csv(root/'counterfactual'/'per_query.csv'); ex=pd.read_csv(root/'exact'/'per_query.csv')
    n=metas[3]['n_queries']; assert n==metas[4]['n_queries']==cf.query_id.nunique()==ex.query_id.nunique()
    assert len(cf)==21*n and len(ex)==6*n
    assert not cf.duplicated(['query_id','method','source_model']).any() and not ex.duplicated(['query_id','source_model']).any()
    assert set(cf.query_id)==set(ex.query_id)
    merged=cf[cf.method.isin(['joint_robust','constrained_random'])].merge(ex,on=['query_id','source_model'],suffixes=('_random','_exact'),validate='one_to_one')
    assert len(merged)==6*n and (merged.found_random==merged.random_found).all() and (merged.extra==merged.found_exact-merged.found_random).all() and (merged.extra>=0).all()
    assert np.allclose(merged.cost_random,merged.random_cost,equal_nan=True)
    both=merged[merged.found_random.eq(1)]; assert (both.cost_exact<=both.cost_random+1e-8).all()
    rng=np.random.default_rng(2026); boot=rng.integers(n,size=(5000,n))
    def ci(x):
        x=np.asarray(x,dtype=float); lo,hi=np.quantile(x[boot].mean(axis=1),[.025,.975]); return float(x.mean()),float(lo),float(hi)
    stats=[]
    for model in MODELS+['joint']:
        g=ex[ex.source_model.eq(model)].sort_values('query_id')
        for metric in ['random_found','found','extra']:
            value,lo,hi=ci(g[metric]); stats.append(dict(model=model,metric=metric,estimate=value,ci_low=lo,ci_high=hi,n=n))
    for model in MODELS:
        g=cf[cf.source_model.eq(model)].pivot(index='query_id',columns='method',values='found')
        for metric,x in [('flip_loss_unconstrained_minus_constrained',g.unconstrained_random-g.constrained_random)]:
            value,lo,hi=ci(x); stats.append(dict(model=model,metric=metric,estimate=value,ci_low=lo,ci_high=hi,n=n))
    joint=cf[cf.method.eq('joint_robust')].set_index('query_id').sort_index()
    union=cf[cf.method.eq('constrained_random')].groupby('query_id').robust_feasible.max().reindex(joint.index)
    value,lo,hi=ci(joint.found-union); stats.append(dict(model='joint',metric='joint_minus_union_single_selections',estimate=value,ci_low=lo,ci_high=hi,n=n))
    pd.DataFrame(stats).to_csv(out/'confidence_intervals.csv',index=False)
    costs=both.assign(cost_reduction=both.cost_random-both.cost_exact).groupby('source_model').cost_reduction.agg(['size','mean','median',lambda x:(x>0).sum()]); costs.columns=['shared_successes','mean_reduction','median_reduction','lower_cost_count']; costs.to_csv(out/'paired_costs.csv')
    cl.to_csv(out/'classification.csv',index=False)
    images=[]
    def save(name,title):
        plt.tight_layout(); plt.savefig(out/(name+'.png'),dpi=220,bbox_inches='tight'); plt.savefig(out/(name+'.svg'),bbox_inches='tight'); plt.close(); images.append((name,title))
    def grouped(frame,columns,labels,title,ylabel,limit=None):
        ax=frame[columns].plot.bar(figsize=(9,4.5),rot=0,width=.8); ax.legend(labels,loc='best'); ax.set_title(title); ax.set_ylabel(ylabel); ax.set_xlabel(''); ax.grid(axis='y',alpha=.2)
        if limit: ax.set_ylim(*limit)
        return ax
    for metric,title in [('f1','Positive-class F1'),('roc_auc','ROC-AUC'),('accuracy','Accuracy')]:
        p=cl.pivot(index='model',columns='features',values=metric).reindex(MODELS); p.index=[LABEL[m] for m in p.index]
        grouped(p,['six','full'],['6 features','14 features'],title+' on official test set',title,(0,1)); save('classification_'+metric,title+' by feature set')
    s=cf.groupby(['method','source_model'])[['found','feasible','robust_feasible']].mean().reset_index()
    for metric,title in [('found','Prediction-flip coverage'),('feasible','Hard-feasible coverage')]:
        p=s.pivot(index='source_model',columns='method',values=metric).reindex(MODELS)*100; p.index=[LABEL[m] for m in p.index]
        grouped(p,METHODS,SHORT,title+' (all queries)', 'Coverage (%)',(0,105)); save('cf_'+metric,title)
    p=ex.groupby('source_model')[['random_found','found']].mean().reindex(MODELS+['joint'])*100; p.index=[LABEL[m] for m in p.index]
    grouped(p,['random_found','found'],['Random constrained','Exhaustive constrained'],'Coverage: random versus exhaustive','Coverage (%)',(0,50)); save('exact_coverage','Random versus exhaustive coverage')
    for method in ['unconstrained_random','constrained_random']:
        p=cf[cf.method.eq(method)].groupby('source_model')[['flip_'+m for m in MODELS]].mean().reindex(MODELS)*100
        fig,ax=plt.subplots(figsize=(6,5)); im=ax.imshow(p,vmin=0,vmax=100,cmap='viridis'); fig.colorbar(im,ax=ax,label='Coverage (%)'); ax.set_xticks(range(5),[LABEL[m] for m in MODELS]); ax.set_yticks(range(5),[LABEL[m] for m in MODELS]); ax.set_xlabel('Evaluated model'); ax.set_ylabel('Source model'); ax.set_title(method.replace('_',' ').title()+' transfer (all queries)')
        for i in range(5):
            for j in range(5): ax.text(j,i,f'{p.iloc[i,j]:.1f}',ha='center',va='center',color='white' if p.iloc[i,j]<55 else 'black')
        save('transfer_'+method,method.replace('_',' ').title()+' transferability')
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    for ax,folder,title in zip(axes,['mlp','transformer'],['MLP','FT-Transformer-style']):
        h=pd.read_csv(root/folder/'history.csv')
        for tag,g in h.groupby('features'):
            ax.plot(g.epoch,g.val_loss,label=f'{tag}: validation'); ax.plot(g.epoch,g.train_loss,'--',label=f'{tag}: training')
        ax.set_title(title); ax.set_xlabel('Epoch'); ax.set_ylabel('BCE'); ax.legend(fontsize=8); ax.grid(alpha=.2)
    save('training_curves','Neural-network training and validation losses')
    fig,ax=plt.subplots(figsize=(8,4.5)); ax.boxplot([both[both.source_model.eq(m)].cost_random-both[both.source_model.eq(m)].cost_exact for m in MODELS+['joint']]); ax.set_xticks(range(1,7),[LABEL[m] for m in MODELS+['joint']]); ax.set_ylabel('Random cost minus exact cost (units)'); ax.set_title('Cost reduction on shared successes only'); ax.grid(axis='y',alpha=.2); save('paired_cost_reduction','Paired effort-cost reductions')
    pred_paths=[root/f/'test_predictions.csv' for f in folders[:3]]
    if all(p.exists() for p in pred_paths):
        from sklearn.metrics import roc_curve,auc
        frames=[pd.read_csv(p) for p in pred_paths]
        for p in frames[1:]: assert frames[0][['row_id','target']].equals(p[['row_id','target']])
        p=frames[0].merge(frames[1].drop(columns='target'),on='row_id',validate='one_to_one').merge(frames[2].drop(columns='target'),on='row_id',validate='one_to_one')
        fig,ax=plt.subplots(figsize=(6,5))
        for m in MODELS:
            x,y,_=roc_curve(p.target,p['full_'+m]); ax.plot(x,y,label=f'{LABEL[m]} ({auc(x,y):.3f})')
        ax.plot([0,1],[0,1],'--'); ax.set(xlabel='False positive rate',ylabel='True positive rate',title='Full-feature ROC curves'); ax.legend(); save('roc_curves','ROC curves on official test set')
    else: print('ROC curves skipped: test_predictions.csv not available for every model family.')
    cases=json.loads((root/'counterfactual'/'cases.json').read_text(encoding='utf8')); selected=next(c for c in cases if c['method']=='joint_robust'); blocked=next(c for c in cases if c['query_id']==selected['query_id'] and not c['feasible'])
    def case(c):
        changes=''.join(f'<tr><td>{html.escape(f)}</td><td>{html.escape(v["before"])}</td><td>{html.escape(v["after"])}</td></tr>' for f,v in c['changes'].items()); scores=''.join(f'<tr><td>{LABEL[m]}</td><td>{c["original_scores"][m]:.3f}</td><td>{c["candidate_scores"][m]:.3f}</td></tr>' for m in MODELS)
        return f'<h3>{html.escape(c["method"])} | query {c["query_id"]}</h3><p>Hard-policy compliant: {c["feasible"]}; caution flag: {c["caution"]}; blocking reason: {html.escape(c["reason"])}</p><table><tr><th>Feature</th><th>Before</th><th>After</th></tr>{changes}</table><table><tr><th>Model</th><th>Before</th><th>After</th></tr>{scores}</table>'
    gallery=''.join(f'<section><h2>{html.escape(title)}</h2><img src="{name}.png" alt="{html.escape(title)}"></section>' for name,title in images)
    (out/'index.html').write_text(f'''<!doctype html><html lang="en"><meta charset="utf-8"><title>CS725 Research Results</title><style>body{{font:16px system-ui;max-width:1000px;margin:auto;padding:24px;color:#202a35}}section{{margin:30px 0}}img{{width:100%;max-width:850px}}table{{border-collapse:collapse;margin:16px 0}}td,th{{border:1px solid #ccc;padding:8px;text-align:left}}small{{color:#555}}</style><h1>Actionable counterfactuals: experimental results</h1><p>UCI Adult | {n:,} common-negative queries | 5 fixed classifiers | threshold 0.5</p><p>Hard-policy compliance is not real-world feasibility or causality. Costs are illustrative. Exhaustive coverage applies only to the defined finite action space. Transfer rates use all queries, including search failures. Bootstrap intervals are conditional on the fixed models and sampled cohort.</p>{gallery}<section><h2>One real held-out profile</h2>{case(blocked)}{case(selected)}</section><p><a href="confidence_intervals.csv">Bootstrap confidence intervals</a> · <a href="paired_costs.csv">Paired costs</a></p></html>''',encoding='utf8')
    print(f'Wrote {len(images)} figures (PNG + SVG), tables and dashboard to {out.resolve()}')

if __name__=='__main__': main()
