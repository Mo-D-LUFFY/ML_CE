"""Research V3: equal-weight ensemble, policy sensitivity and local live demo.
Preserves all V2 results. No training or fitting on the official test set.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from run_counterfactuals import (ROOT, ALL, NUM, NUMI, NAMES, H,E,ED,JOB,MISSING,
    load, clean, models, probabilities, assess, mixed_distance, rank)
from run_exact import enumerate_candidates
from train_classical import scores

LABELS=NAMES+['ensemble']
POLICIES={
    'hours_only':'Hours increase only, original V2 bounds',
    'education_only':'Education increase only, conditional',
    'employment_only':'Employment pair change only, conditional',
    'no_overtime':'Hours increases only up to 40; education/job changes permitted',
    'same_workclass':'Keep employer type fixed; occupation/education/hours may change',
    'low_effort':'Expanded policy with effort cost <= 10',
    'no_conditional':'No education/job change; hours increase <= 10',
    'expanded':'Original expanded V2 policy',
}
def with_ensemble(p):
    return np.column_stack([p,p.mean(axis=1)])

def evaluate_ensemble(root,out):
    for tag in ['six','full']:
        arrays=[]
        first=None
        for folder,ns in [('classical',NAMES[:3]),('mlp',['mlp']),('transformer',['ft'])]:
            f=pd.read_csv(root/folder/'test_predictions.csv')
            if first is None: first=f[['row_id','target']]
            elif not first.equals(f[['row_id','target']]): raise ValueError('Test row mismatch')
            arrays.append(f[[tag+'_'+n for n in ns]].to_numpy())
        p=with_ensemble(np.column_stack(arrays))
        rows=[dict(features=tag,model=n,**scores(first.target,p[:,j])) for j,n in enumerate(LABELS)]
        pd.DataFrame(rows).to_csv(out/f'ensemble_{tag}_metrics.csv',index=False)
        first.assign(ensemble=p[:,-1]).to_csv(out/f'ensemble_{tag}_predictions.csv',index=False)
        print(pd.DataFrame(rows)[['features','model','accuracy','f1','roc_auc']].to_string(index=False),flush=True)

def masks(a,x,feasible,caution,cost):
    dh=a[:,H].astype(float)-float(x[H])
    ec=(a[:,[ED,E]]!=x[[ED,E]]).any(axis=1)
    jc=(a[:,JOB]!=x[JOB]).any(axis=1)
    hc=dh!=0
    return {k:feasible&v for k,v in {
        'hours_only':hc&~ec&~jc,
        'education_only':ec&~hc&~jc,
        'employment_only':jc&~hc&~ec,
        'no_overtime':(~hc)|(a[:,H].astype(float)<=40),
        'same_workclass':a[:,JOB[0]]==x[JOB[0]],
        'low_effort':cost<=10,
        'no_conditional':~caution,
        'expanded':np.ones(len(a),dtype=bool)}.items()}

class Engine:
    def __init__(self,root,device):
        (train,test),self.hashes=load(ROOT.parent/'cs725_starter'/'data')
        for folder in ['classical','mlp','transformer']:
            m=json.loads((root/folder/'metadata.json').read_text(encoding='utf8'))
            if m['sha256']!=self.hashes: raise ValueError('Dataset hash mismatch: '+folder)
        self.tr,self.te=clean(train),clean(test)
        self.ms=models(root,device)
        self.edu=set(zip(self.tr[:,ED],self.tr[:,E]))
        self.jobs={tuple(v) for v in self.tr[:,JOB]}
        v=self.tr[:,NUMI].astype(float)
        self.scale=np.maximum(np.quantile(v,.95,axis=0)-np.quantile(v,.05,axis=0),1)
    def predict(self,a):
        return with_ensemble(probabilities(self.ms,a))
    def space(self,x):
        a=enumerate_candidates(x,self.edu,self.jobs)
        feasible,caution,cost,edits,_=assess(a,x,self.edu,self.jobs)
        if not feasible.all(): raise ValueError('Policy enumeration failed')
        p=self.predict(a) if len(a) else np.empty((0,6))
        dist=mixed_distance(a,x,self.scale)
        return a,p,cost,dist,edits,caution,masks(a,x,feasible,caution,cost)
    def options(self,x,policy,target):
        before=self.predict(x[None,:])[0]
        if target=='joint':
            if (before[:5]>=.5).all(): return before,'ALREADY_SATISFIED',None
        elif before[LABELS.index(target)]>=.5:
            return before,'ALREADY_SATISFIED',None
        a,p,c,d,e,warning,allowed=self.space(x)
        valid=(p[:,:5]>=.5).all(axis=1) if target=='joint' else p[:,LABELS.index(target)]>=.5
        k=rank(np.flatnonzero(allowed[policy]&valid),d,c,True)
        if k is None: return before,'NO_CANDIDATE_IN_POLICY',None
        return before,'CAUTION' if warning[k] else 'PASS',dict(
            after=p[k],cost=float(c[k]),edits=int(e[k]),
            changes=[(f,str(x[i]),str(a[k,i])) for i,f in enumerate(ALL) if a[k,i]!=x[i]])

def benchmark(engine,root,out,n):
    old=pd.read_csv(root/'counterfactual'/'per_query.csv')
    ids=old.query_id.drop_duplicates().to_numpy()[:n]
    if len(ids)!=n: raise ValueError('Requested more queries than V2 has')
    if not (engine.predict(engine.te[ids])<.5).all(): raise ValueError('Cohort mismatch')
    rows=[]
    def save():
        f=pd.DataFrame(rows)
        f.to_csv(out/'policy_per_query.csv',index=False)
        f.groupby(['policy','target'],sort=False).agg(queries=('found','size'),
            coverage=('found','mean'),median_cost=('cost','median'),caution_coverage=('caution','mean'),
            mean_candidates=('candidates','mean')).reset_index().to_csv(out/'policy_summary.csv',index=False)
    for i,q in enumerate(ids):
        a,p,c,d,e,warning,allowed=engine.space(engine.te[q])
        for name,mask in allowed.items():
            for target in LABELS+['joint']:
                valid=(p[:,:5]>=.5).all(axis=1) if target=='joint' else p[:,LABELS.index(target)]>=.5
                k=rank(np.flatnonzero(mask&valid),d,c,True)
                rows.append(dict(query_id=int(q),policy=name,target=target,found=int(k is not None),
                    cost=float(c[k]) if k is not None else np.nan,caution=int(warning[k]) if k is not None else 0,
                    candidates=int(mask.sum())))
        if (i+1)%25==0 or i==0: print(f'Policy queries {i+1}/{n}',flush=True); save()
    save()
    (out/'metadata.json').write_text(json.dumps(dict(dataset='UCI Adult official split',sha256=engine.hashes,
        queries=n,policies=POLICIES,threshold=.5,ensemble='fixed unweighted mean of five scores',
        interpretation='Policies filter the same expanded exhaustive action space; no_overtime allows unchanged hours above 40; joint is all five base models. Costs illustrative; no causal claims.'),indent=2),encoding='utf8')

PAGE="""<!doctype html><html lang="en"><meta charset="utf-8"><title>Counterfactual research lab</title>
<style>body{font:16px system-ui;background:#f3f5f8;color:#182332;margin:0}main{max-width:1050px;margin:auto;padding:28px}section{background:white;border:1px solid #dce2e9;border-radius:12px;padding:22px;margin:18px 0}h1{font-size:29px}form .grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:13px}label{display:block;font-size:13px;font-weight:600}input,select{box-sizing:border-box;width:100%;padding:9px;margin-top:5px;border:1px solid #aab6c4;border-radius:5px;background:white}button{background:#163b64;color:white;border:0;padding:12px 18px;border-radius:6px;margin-top:18px;cursor:pointer}table{border-collapse:collapse;width:100%}td,th{padding:9px;text-align:left;border-bottom:1px solid #ddd}.notice{color:#8a4f00}.score{font-variant-numeric:tabular-nums}a{color:#1d5287}</style>
<main><h1>Actionable counterfactual research lab</h1><p>Historical UCI Adult benchmark • five trained models + soft-voting ensemble • live local inference.</p>
<p class="notice">Research demonstration only. These scores are not salary forecasts or causal advice. Conditional actions are not verified. Do not enter identifiable personal information.</p>
{% if report %}<p><a href="/report/">Open experimental results dashboard</a></p>{% endif %}
<section><h2>Custom profile</h2><p>Defaults are one held-out census profile. Census weight is metadata, not personal wealth.</p>
<form method="post"><div class="grid">{% for f in fields %}<label>{{f.name}}
{% if f.options is not none %}<select name="{{f.name}}">{% for v in f.options %}<option {% if v==f.value %}selected{% endif %}>{{v}}</option>{% endfor %}</select>
{% else %}<input type="number" name="{{f.name}}" value="{{f.value}}" min="{{f.min}}" max="{{f.max}}" step="1" required>{% endif %}</label>{% endfor %}</div>
<div class="grid" style="margin-top:16px"><label>Policy<select name="policy">{% for k,v in policies.items() %}<option value="{{k}}" {% if k==policy %}selected{% endif %}>{{v}}</option>{% endfor %}</select></label>
<label>Target<select name="target">{% for n in targets %}<option {% if n==target %}selected{% endif %}>{{n}}</option>{% endfor %}</select></label></div><button>Predict and search</button></form>
{% if error %}<p class="notice">{{error}}</p>{% endif %}</section>
{% if before is not none %}<section><h2>Model outputs</h2><p>Decision threshold = 0.5. Status: <strong>{{status}}</strong></p><table><tr><th>Model</th><th>Original score</th><th>Original class</th>{% if result %}<th>Candidate score</th><th>Candidate class</th>{% endif %}</tr>{% for name,p,q in score_rows %}<tr><td>{{name}}</td><td class="score">{{"%.2f"|format(p*100)}}%</td><td>{{" >50K" if p>=0.5 else "≤50K"}}</td>{% if result %}<td class="score">{{"%.2f"|format(q*100)}}%</td><td>{{" >50K" if q>=0.5 else "≤50K"}}</td>{% endif %}</tr>{% endfor %}</table>
{% if result %}<h3>Minimum-cost policy candidate</h3><p>Illustrative cost: {{result.cost}} units. Edited columns: {{result.edits}}.</p><table><tr><th>Feature</th><th>Original</th><th>Modified</th></tr>{% for f,a,b in result.changes %}<tr><td>{{f}}</td><td>{{a}}</td><td>{{b}}</td></tr>{% endfor %}</table>{% endif %}</section>{% endif %}</main></html>"""

def serve(engine,root,port):
    from flask import Flask,request,render_template_string,send_from_directory
    app=Flask(__name__)
    test=engine.te
    scores0=engine.predict(test)
    default=test[np.flatnonzero((scores0[:,:5]<.5).all(axis=1))[0]].copy()
    @app.route('/report/')
    def report(): return send_from_directory(root/'report','index.html')
    @app.route('/report/<path:name>')
    def asset(name): return send_from_directory(root/'report',name)
    @app.route('/',methods=['GET','POST'])
    def index():
        x=default.copy(); error=None; before=None; result=None; status=None
        policy=request.form.get('policy','expanded'); target=request.form.get('target','joint')
        fields=[]
        try:
            if policy not in POLICIES or target not in LABELS+['joint']: raise ValueError('Invalid policy or target')
            if request.method=='POST':
                for i,f in enumerate(ALL):
                    raw=request.form[f]
                    if f in NUM:
                        value=int(raw); lo=int(engine.tr[:,i].min()); hi=int(engine.tr[:,i].max())
                        if not lo<=value<=hi: raise ValueError(f'{f} outside training range')
                        x[i]=value
                    else:
                        if raw not in {str(v) for v in engine.tr[:,i]}|{str(default[i])}: raise ValueError(f'Unknown {f}')
                        x[i]=raw
                if (x[ED],x[E]) not in engine.edu: raise ValueError('Education and education-num must match')
                if tuple(x[JOB]) not in engine.jobs: raise ValueError('Workclass/occupation pair not observed in training')
                before,status,result=engine.options(x,policy,target)
        except (ValueError,KeyError) as exc: error=str(exc)
        for i,f in enumerate(ALL):
            if f in NUM: fields.append(dict(name=f,value=int(x[i]),options=None,min=int(engine.tr[:,i].min()),max=int(engine.tr[:,i].max())))
            else: fields.append(dict(name=f,value=str(x[i]),options=sorted({str(v) for v in engine.tr[:,i]}|{str(default[i])})))
        score_rows=[] if before is None else [(n,float(before[j]),float(result['after'][j]) if result else None) for j,n in enumerate(LABELS)]
        return render_template_string(PAGE,fields=fields,policies=POLICIES,policy=policy,target=target,
            targets=LABELS+['joint'],before=before,result=result,error=error,status=status,score_rows=score_rows,
            report=(root/'report'/'index.html').exists())
    app.run(host='127.0.0.1',port=port,debug=False)

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('mode',choices=['ensemble','policies','demo'])
    ap.add_argument('--results',type=Path,default=ROOT/'results')
    ap.add_argument('--queries',type=int,default=1000)
    ap.add_argument('--port',type=int,default=5000)
    ap.add_argument('--device',choices=['cpu','cuda','auto'],default='auto')
    args=ap.parse_args(); root=args.results.resolve(); out=root/'v3'; out.mkdir(parents=True,exist_ok=True)
    if args.mode=='ensemble': evaluate_ensemble(root,out); return
    if args.queries<1: ap.error('queries must be positive')
    device=('cuda' if torch.cuda.is_available() else 'cpu') if args.device=='auto' else args.device
    torch.set_num_threads(4)
    engine=Engine(root,device)
    if args.mode=='policies': benchmark(engine,root,out,args.queries)
    else: serve(engine,root,args.port)
if __name__=='__main__': main()
