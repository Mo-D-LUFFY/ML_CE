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
        if not len(a):
            empty=np.empty(0)
            return a,np.empty((0,6)),empty,empty,empty.astype(int),empty.astype(bool),{k:empty.astype(bool) for k in POLICIES}
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

# Live settings are separate from the fixed eight-policy research benchmark.
LIVE_RULES={
    'hours':('Allow working-hour changes','At most +15 hours/week, with a final maximum of 65.'),
    'education':('Allow further education','Increase by one or two Adult education levels; conditional.'),
    'job':('Allow employment changes','Only observed workclass/occupation combinations; conditional.'),
    'overtime':('Avoid new overtime','If hours change, the new value must not exceed 40.'),
    'employer':('Keep employer type','Occupation may change, but workclass must stay the same.'),
    'effort':('Limit effort to 10 units','Illustrative cost cap, not money.'),
    'caution':('Avoid conditional changes','Reject education/job changes and hours increases above 10.'),
}
DEFAULT_RULES={'hours','education','job'}
MODEL_NAMES=dict(zip(LABELS,['Logistic Regression','Random Forest','XGBoost','Deep MLP','FT-Transformer','Ensemble (equal-weight soft voting)']))
FIELD_LABELS={'age':'Age (years)','education':'Highest education','education-num':'Education code',
    'hours-per-week':'Working hours per week','capital-gain':'Recorded capital gain (USD)',
    'capital-loss':'Recorded capital loss (USD)','fnlwgt':'Census sampling weight',
    'workclass':'Employer type','occupation':'Occupation','marital-status':'Marital status',
    'relationship':'Household relationship','race':'Recorded race category','sex':'Recorded sex category',
    'native-country':'Native country'}
FIELD_HELP={'age':'Age in years at the time of the record.','education':'Choose the qualification; its numeric code is filled automatically.',
    'hours-per-week':'Total usual hours in a week, not hours per day.',
    'capital-gain':'Recorded investment/asset gains, not annual salary. Enter 0 when none are recorded.',
    'capital-loss':'Recorded investment/asset losses, as a nonnegative amount. Enter 0 when none are recorded.',
    'fnlwgt':'Survey metadata: how many people a census record represents. For hypothetical profiles, leave the training median unless you have a known weight.',
    'workclass':'Type of employer or self-employment.','occupation':'Choose an occupation observed with the selected employer type.',
    'marital-status':'Historical census category.','relationship':'Relationship to household head, not employment.',
    'race':'Historical model input; never suggested as an action.','sex':'Historical model input; never suggested as an action.',
    'native-country':'Recorded country of origin, not current location.'}

def live_options(engine,x,rules,target):
    before=engine.predict(x[None,:])[0]
    valid0=(before[:5]>=.5).all() if target=='joint' else before[LABELS.index(target)]>=.5
    if valid0: return before,'Already meets the selected prediction target',None,0
    a=enumerate_candidates(x,engine.edu,engine.jobs)
    if not len(a): return before,'No candidate in the defined action space',None,0
    feasible,caution,cost,edits,_=assess(a,x,engine.edu,engine.jobs)
    allowed=feasible.copy()
    if 'hours' not in rules: allowed &= a[:,H]==x[H]
    if 'education' not in rules: allowed &= (a[:,[ED,E]]==x[[ED,E]]).all(axis=1)
    if 'job' not in rules: allowed &= (a[:,JOB]==x[JOB]).all(axis=1)
    if 'overtime' in rules: allowed &= (a[:,H]==x[H])|(a[:,H].astype(float)<=40)
    if 'employer' in rules: allowed &= a[:,JOB[0]]==x[JOB[0]]
    if 'effort' in rules: allowed &= cost<=10
    if 'caution' in rules: allowed &= ~caution
    a=a[allowed]; c=cost[allowed]; warning=caution[allowed]; edits=edits[allowed]
    if not len(a): return before,'No candidate permitted by active controls',None,0
    p=engine.predict(a)
    dist=mixed_distance(a,x,engine.scale)
    valid=(p[:,:5]>=.5).all(axis=1) if target=='joint' else p[:,LABELS.index(target)]>=.5
    ix=np.flatnonzero(valid)
    if not len(ix): return before,'No prediction-flipping candidate within active policy',None,len(a)
    # Use distinct observed cost levels, not arbitrary cost multipliers or synthetic candidates.
    levels=np.unique(c[ix])
    tiers=[('minimum','Minimum cost',levels[0])]
    if len(levels)>=3:
        tiers.append(('medium','Medium cost',levels[len(levels)//2]))
    if len(levels)>=2:
        tiers.append(('maximum','Maximum cost',levels[-1]))
    def describe(key,title,k):
        changes=[(FIELD_LABELS[f],str(x[i]),str(a[k,i])) for i,f in enumerate(ALL) if a[k,i]!=x[i]]
        notes=[]
        if a[k,H]!=x[H] and float(a[k,H])-float(x[H])>10: notes.append('Large increase in weekly working hours')
        if (a[k,[ED,E]]!=x[[ED,E]]).any(): notes.append('Education requires time, resources and access')
        if (a[k,JOB]!=x[JOB]).any(): notes.append('Employment opportunity and qualifications are unverified')
        return dict(tier=key,title=title,after=p[k],cost=float(c[k]),edits=int(edits[k]),
            changes=changes,notes=notes,caution=bool(warning[k]),
            target_score=float(p[k,:5].min() if target=='joint' else p[k,LABELS.index(target)]),
            positive_count=int((p[k,:5]>=.5).sum()),
            scores=[(MODEL_NAMES[n],float(p[k,j])) for j,n in enumerate(LABELS)])
    options=[]
    for key,title,level in tiers:
        eligible=ix[c[ix]==level]
        # Same minimum-distance tie-break at every effort level.
        k=rank(eligible,dist,c,True)
        options.append(describe(key,title,k))
    result=dict(options[0],alternatives=options,valid_count=int(len(ix)),cost_levels=int(len(levels)))
    return before,'CAUTION — conditional action' if result['caution'] else 'PASS — hard rules satisfied',result,len(a)


PAGE="""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Counterfactual Research Lab</title>
<style>
*{box-sizing:border-box}body{font:15px system-ui;background:#f2f5f9;color:#172b40;margin:0}main{max-width:1100px;margin:auto;padding:28px}h1{font-size:30px;margin:8px 0}h2{font-size:21px;margin:0 0 14px}h3{font-size:16px}p{line-height:1.55}section{background:white;border:1px solid #dbe3ed;border-radius:12px;padding:22px;margin:18px 0}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(230px,1fr));gap:18px}.field label{display:block;font-weight:650}input:not([type=checkbox]),select{width:100%;padding:10px;border:1px solid #adbacb;border-radius:6px;background:white;margin:6px 0;font:inherit}small{display:block;color:#52677c;line-height:1.4}button,.linkbtn{font:inherit;font-weight:600;background:#174f78;color:white;border:0;border-radius:6px;padding:11px 16px;cursor:pointer;text-decoration:none;display:inline-block}a{color:#174f78}nav{display:flex;gap:18px;flex-wrap:wrap}summary{cursor:pointer;font-weight:650;margin:12px 0}.rule{border:1px solid #dbe3ed;border-radius:7px;padding:12px;display:flex;gap:9px;align-items:flex-start}.rule input{margin-top:4px}.rule small{margin-top:4px}.notice{border-left:3px solid #b7791f;padding-left:12px;color:#79501c}.muted{color:#52677c}.tablewrap{overflow:auto}table{border-collapse:collapse;width:100%;font-size:14px}td,th{padding:10px;text-align:left;border-bottom:1px solid #e2e8ef}th{background:#f5f8fb}.track{height:9px;background:#e3eaf1;border-radius:5px;min-width:70px;overflow:hidden}.bar{height:100%;background:#2b719e}.positive{background:#147d69}.metric{font-size:25px;font-weight:700}.meta{display:flex;flex-wrap:wrap;gap:25px}.pill{font-size:12px;background:#e9eff5;padding:4px 8px;border-radius:20px}.error{color:#9e2828;font-weight:600}@media print{body{background:white}main{padding:0;max-width:none}#form,header,nav,.no-print{display:none!important}section{border:0;padding:0}details{display:block}table{break-inside:avoid}}
header{background:#17364f;color:#fff;border-radius:14px;padding:24px 28px;box-shadow:0 5px 20px #17364f14}header .muted,header a{color:#d8e6f2}header .notice{color:#ffdf9c;border-color:#ffdf9c}header h1{letter-spacing:-.7px}section{box-shadow:0 3px 16px #17364f08}.field input:focus,.field select:focus{outline:2px solid #73a8cb;outline-offset:1px}.rule:has(input:checked){background:#f2f7fb;border-color:#a6c4db}.rule input{accent-color:#174f78}.actions{display:flex;flex-wrap:wrap;gap:10px;align-items:center;margin-top:16px}.actions button{margin:0}.actions button:hover{filter:brightness(.95)}.pick-pass{background:#e9f6ef;color:#16643f;border:1px solid #83c5a2}.pick-fail{background:#fff0ef;color:#a32c2c;border:1px solid #e5a29c}.outcome{border-radius:10px;padding:19px 22px;border:1px solid;display:flex;align-items:flex-start;gap:16px;margin:0 0 18px}.outcome .symbol{font-size:29px;font-weight:700;line-height:1.1}.outcome h3{font-size:21px;margin:0 0 4px}.outcome p{margin:0}.outcome.green{background:#edf8f1;border-color:#a5d7b7;color:#16643f}.outcome.yellow{background:#fff8e5;border-color:#ead18b;color:#79500c}.outcome.red{background:#fff0ef;border-color:#e7ada8;color:#9e2e2b}.meta{display:grid;grid-template-columns:repeat(auto-fit,minmax(155px,1fr));gap:12px;margin-bottom:18px}.meta>div{border:1px solid #dbe3ed;background:#f8fafc;border-radius:9px;padding:14px}.tag{display:inline-block;padding:3px 8px;border-radius:5px;font-weight:650;font-size:12px}.tag.yes{color:#16643f;background:#e9f6ef}.tag.no{color:#9e2e2b;background:#fff0ef}.bar:not(.positive){background:#b95d56}.sample{padding:10px 14px;background:#eef5fb;border-radius:6px;color:#174f78}.stale{padding:10px;border:1px solid #e6cf9b;background:#fff8e5;color:#79500c;border-radius:6px}button:focus-visible,a:focus-visible{outline:3px solid #73a8cb;outline-offset:3px}@media(max-width:600px){main{padding:12px}section,header{padding:16px}.actions button{width:100%}.outcome{padding:15px}.outcome h3{font-size:18px}}@media print{.outcome,.tag,.bar{print-color-adjust:exact;-webkit-print-color-adjust:exact}}
.policy-summary{display:grid;grid-template-columns:repeat(auto-fit,minmax(230px,1fr));gap:12px;margin:12px 0 22px}.policy-card{background:#f8fbfe;border:1px solid #d8e3ee;border-radius:9px;padding:13px 14px;min-width:0;break-inside:avoid}.policy-card.enabled{border-color:#9acbb2;background:#f3faf6}.policy-head{display:flex;align-items:flex-start;justify-content:space-between;gap:10px;margin-bottom:6px}.policy-name{font-weight:650;line-height:1.35;overflow-wrap:anywhere}.policy-state{flex-shrink:0;font-size:11px;font-weight:750;white-space:nowrap;border-radius:5px;padding:4px 8px}.state-on{background:#dcefe4;color:#16643f}.state-off{background:#e6ebf0;color:#536476}.policy-desc{font-size:13px;line-height:1.45;color:#52677c}.tier-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,280px),1fr));gap:14px;align-items:start}.tier-card{border:1px solid #d8e3ee;background:#fff;border-radius:10px;padding:16px;min-width:0;break-inside:avoid}.tier-card.minimum{border-top:4px solid #16836b}.tier-card.medium{border-top:4px solid #c29135}.tier-card.maximum{border-top:4px solid #64748b}.tier-card h4{margin:0 0 6px;font-size:17px}.tier-card .cost{font-size:25px;font-weight:750;line-height:1.3}.tier-card .tier-metrics{display:grid;grid-template-columns:1fr 1fr;gap:9px;margin:12px 0}.tier-card .tier-metrics div{background:#f4f7fa;padding:9px;border-radius:6px}.tier-card table{font-size:12px;table-layout:fixed}.tier-card td,.tier-card th{padding:7px;overflow-wrap:anywhere}.tier-card .notice{font-size:13px}.tier-card details{margin-top:10px}.tier-card details summary{font-size:13px}.tier-label{display:inline-block;background:#edf2f7;border-radius:4px;padding:4px 8px;font-size:12px;margin-bottom:9px}.tier-card.minimum .tier-label{background:#e5f4eb;color:#16643f}.comparison-note{font-size:13px;color:#52677c}.policy-summary-title{display:flex;align-items:center;justify-content:space-between;gap:8px;flex-wrap:wrap}.policy-summary-title h3{margin:8px 0}@media print{.tier-grid{display:block}.tier-card{margin:12px 0}.policy-summary{grid-template-columns:repeat(2,minmax(0,1fr))}}
</style></head><body><main>
<header><p class="muted">CS725 / Research V3</p><h1>Counterfactual Research Lab</h1><p>Adult historical US census benchmark · Five trained classifiers and their soft-voting ensemble</p>
<nav><a href="#form">Profile input</a>{% if before is not none %}<a href="#profile-report">Current profile report</a>{% endif %}{% if report %}<a href="/report/" target="_blank" rel="noopener">Training &amp; experiment results ↗</a><a href="/report/workflow.html" target="_blank" rel="noopener">Project workflow ↗</a>{% endif %}</nav>
<p class="notice">Research demonstration only. Scores are model outputs, not actual earnings forecasts. Do not enter identifiable information.</p></header>
<section id="form"><h2>1. Enter a hypothetical profile</h2><p class="muted">A held-out example is prefilled. Numbers below are contextualized using unweighted statistics from the actual training split, not recommended personal targets.</p>
<form method="post" id="profile-form">
{% for group,fs in groups %}{% if loop.index==2 %}<details {% if error %}open{% endif %}><summary>{{group}}</summary>{% else %}<h3>{{group}}</h3>{% endif %}
<div class="grid">{% for f in fs %}<div class="field"><label for="{{f.name}}">{{f.label}}</label>
{% if f.options is not none %}<select id="{{f.name}}" name="{{f.name}}">{% for v in f.options %}<option value="{{v}}" {% if v==f.value %}selected{% endif %}>{{ 'Not recorded' if v==missing else v }}</option>{% endfor %}</select>
{% else %}<input id="{{f.name}}" type="number" name="{{f.name}}" value="{{f.value}}" min="{{f.min}}" max="{{f.max}}" step="1" required>{% endif %}
<small>{{f.help}}</small>{% if f.stats %}<small>{{f.stats}}</small>{% endif %}
{% if f.name=='education' %}<small>Derived education code: <strong id="edu-code">{{education_code}}</strong> (read-only)</small>{% endif %}
</div>{% endfor %}</div>{% if loop.index==2 %}</details>{% endif %}{% endfor %}
<h2 style="margin-top:26px">2. Set permitted actions</h2><p class="muted">The three action permissions are enabled by default. Uncheck any action that is unavailable; combine additional restrictions if desired.</p>
<div class="grid">{% for key,(label,desc) in live_rules.items() %}<label class="rule"><input type="checkbox" name="rules" value="{{key}}" {% if key in rules %}checked{% endif %}><span><strong>{{label}}</strong><small>{{desc}}</small></span></label>{% endfor %}</div>
<p class="muted">Always enforced: frozen historical/demographic features, consistent education codes, observed employment pairs and bounded hours. Turning off a restriction does not allow immutable edits.</p>
<div class="field"><label for="target">Prediction objective</label><select name="target" id="target">{% for n,label in targets.items() %}<option value="{{n}}" {% if n==target %}selected{% endif %}>{{label}}</option>{% endfor %}</select></div>
<div class="actions"><button type="submit" name="action" value="analyze">Analyze this profile</button><button type="submit" name="action" value="pick_pass" class="pick-pass" formnovalidate>Pick one that will pass</button><button type="submit" name="action" value="pick_fail" class="pick-fail" formnovalidate>Pick one that will fail</button><a href="/">Reset example</a></div>
<p class="muted">Random examples are held-out records, checked against the selected objective and active controls. The fail button searches for a negative profile with no valid counterfactual under those controls, not merely a negative prediction.</p>
{% if sample_id is not none %}<p class="sample">Random held-out example · test row {{sample_id}} · original recorded attributes retained.</p>{% endif %}
{% if error %}<p class="error">{{error}}</p>{% endif %}</form></section>
{% if before is not none %}<section id="profile-report"><h2>3. Current profile report</h2><p class="muted">Generated from the submitted values, not the historical experiment dashboard.</p>
<div id="stale" class="stale no-print" hidden>Inputs or controls have changed. Analyze again to update this report.</div>
<div class="outcome {{outcome}}" role="status"><span class="symbol" aria-hidden="true">{{'✓' if outcome=='green' else '!' if outcome=='yellow' else '×'}}</span><div><h3>{{outcome_title}}</h3><p>{{outcome_detail}}</p></div></div>
<div class="meta"><div><small>Ensemble score before</small><div class="metric">{{'%.2f'|format(before[-1]*100)}}%</div></div><div><small>Base models already positive</small><div class="metric">{{positive_count}} / 5</div></div><div><small>Permitted candidates evaluated</small><div class="metric">{{candidate_count}}</div></div>{% if result %}<div><small>Ensemble after minimum cost</small><div class="metric">{{'%.2f'|format(result.after[-1]*100)}}%</div></div><div><small>Base models positive · minimum cost</small><div class="metric">{{after_count}} / 5</div></div>{% endif %}</div>
<p><strong>Objective:</strong> {{targets[target]}} · Threshold: 50%.</p><p><strong>Status:</strong> {{status}}</p>
<div class="tablewrap"><table><thead><tr><th>Model</th><th>Original score</th><th>Original class</th>{% if result %}<th>Minimum-cost score</th><th>Minimum-cost class</th>{% endif %}</tr></thead><tbody>{% for name,p,q in score_rows %}<tr><td>{{name}}</td><td>{{'%.2f'|format(p*100)}}%<div class="track"><div class="bar {% if p>=0.5 %}positive{% endif %}" style="width:{{p*100}}%"></div></div></td><td><span class="tag {{'yes' if p>=0.5 else 'no'}}">{{'>50K' if p>=0.5 else '≤50K'}}</span></td>{% if result %}<td>{{'%.2f'|format(q*100)}}%<div class="track"><div class="bar {% if q>=0.5 %}positive{% endif %}" style="width:{{q*100}}%"></div></div></td><td><span class="tag {{'yes' if q>=0.5 else 'no'}}">{{'>50K' if q>=0.5 else '≤50K'}}</span></td>{% endif %}</tr>{% endfor %}</tbody></table></div>
{% if result %}
<h3>Compare counterfactual options</h3>
<p class="comparison-note">{{result.valid_count}} valid candidates across {{result.cost_levels}} distinct effort levels. All displayed options satisfy the selected objective and active hard rules. The medium option uses the middle distinct cost level; maximum cost is a comparison, not a recommendation. More effort need not improve model scores.</p>
{% if result.cost_levels<3 %}<p class="muted">Only {{result.cost_levels}} distinct valid effort level(s) exist under these controls. Duplicate tiers are omitted.</p>{% endif %}
<div class="tier-grid">{% for option in result.alternatives %}
<article class="tier-card {{option.tier}}">
<span class="tier-label">{{'Lowest declared effort' if option.tier=='minimum' else 'Middle effort level' if option.tier=='medium' else 'Highest permitted effort'}}</span>
<h4>{{option.title}} counterfactual</h4>
<div class="cost">{{'%.0f'|format(option.cost)}} <small style="display:inline">units</small></div>
<div class="tier-metrics"><div><small>Selected objective score{% if target=='joint' %} (weakest model){% endif %}</small><strong>{{'%.2f'|format(option.target_score*100)}}%</strong></div><div><small>Ensemble score</small><strong>{{'%.2f'|format(option.after[-1]*100)}}%</strong></div><div><small>Positive base models</small><strong>{{option.positive_count}} / 5</strong></div><div><small>Edited columns</small><strong>{{option.edits}}</strong></div></div>
<span class="tag {{'no' if option.caution else 'yes'}}">{{'CAUTION — conditional' if option.caution else 'PASS — hard rules satisfied'}}</span>
<h4 style="margin-top:15px">Required feature changes</h4>
<table><thead><tr><th>Feature</th><th>Before</th><th>After</th></tr></thead><tbody>{% for f,a,b in option.changes %}<tr><td>{{f}}</td><td>{{a}}</td><td><strong>{{b}}</strong></td></tr>{% endfor %}</tbody></table>
{% for note in option.notes %}<p class="notice">{{note}}</p>{% endfor %}
<details><summary>All six model scores</summary><table><thead><tr><th>Model</th><th>Modified score</th></tr></thead><tbody>{% for name,p in option.scores %}<tr><td>{{name}}</td><td><span class="tag {{'yes' if p>=0.5 else 'no'}}">{{'%.2f'|format(p*100)}}%</span></td></tr>{% endfor %}</tbody></table></details>
</article>{% endfor %}</div>
{% else %}<p class="muted">An already-positive objective needs no change. Otherwise, a missing candidate means none was found in the finite permitted domain; it does not establish real-world impossibility.</p>{% endif %}
<div class="policy-summary-title"><h3>Active policy settings</h3><small>{{rules|length}} of {{live_rules|length}} controls enabled</small></div>
<div class="policy-summary">{% for key,(label,desc) in live_rules.items() %}<article class="policy-card {{'enabled' if key in rules else ''}}"><div class="policy-head"><span class="policy-name">{{label}}</span><span class="policy-state {{'state-on' if key in rules else 'state-off'}}">{{'ON' if key in rules else 'OFF'}}</span></div><div class="policy-desc">{{desc}}</div></article>{% endfor %}</div>
<h3>Submitted profile</h3><div class="tablewrap"><table>{% for f,v in submitted %}<tr><th>{{f}}</th><td>{{v}}</td></tr>{% endfor %}</table></div>
<p class="muted">The ensemble is the mean of five scores. It is not independent evidence. Hard-policy compliance does not verify actual feasibility or causality.</p>
<button type="button" class="no-print" onclick="window.print()">Print / Save profile report as PDF</button>
<button type="button" class="no-print" id="download">Download profile report JSON</button>
</section>{% endif %}
<script>
const edu={{edu_map|tojson}}, jobs={{job_map|tojson}};
const ed=document.getElementById('education'), wc=document.getElementById('workclass'), occ=document.getElementById('occupation');
function updateEdu(){document.getElementById('edu-code').textContent=edu[ed.value];}
function updateJob(){const previous=occ.value;occ.replaceChildren();for(const v of (jobs[wc.value]||[])){const o=new Option(v==={{missing|tojson}}?'Not recorded':v,v);occ.add(o);}if(Array.from(occ.options).some(o=>o.value===previous))occ.value=previous;}
ed.addEventListener('change',updateEdu);wc.addEventListener('change',updateJob);updateEdu();updateJob();
{% if before is not none %}
const reportData={{payload|tojson}};
document.getElementById('download').addEventListener('click',()=>{const url=URL.createObjectURL(new Blob([JSON.stringify(reportData,null,2)],{type:'application/json'}));const a=document.createElement('a');a.href=url;a.download='profile_report.json';a.click();URL.revokeObjectURL(url);});
document.getElementById('profile-form').addEventListener('input',()=>{document.getElementById('stale').hidden=false;});
document.getElementById('profile-report').scrollIntoView({block:'start'});
{% endif %}
</script></main></body></html>"""

def create_app(engine,root):
    from flask import Flask,request,render_template_string,send_from_directory
    app=Flask(__name__)
    scores0=engine.predict(engine.te)
    eligible=np.flatnonzero((scores0[:,:5]<.5).all(axis=1))
    idx=14381 if 14381 in eligible else int(eligible[0]) if len(eligible) else 0
    default=engine.te[idx].copy()
    edu_map={str(a):int(b) for a,b in engine.edu}
    if len(edu_map)!=len(engine.edu): raise ValueError('Education code map is not one-to-one')
    job_map={}
    for a,b in sorted(engine.jobs,key=lambda v:tuple(map(str,v))): job_map.setdefault(str(a),[]).append(str(b))
    if tuple(default[JOB]) not in engine.jobs:
        default[JOB]=next(iter(sorted(engine.jobs,key=lambda v:tuple(map(str,v)))))
    cats={f:sorted({str(v) for v in engine.tr[:,i]}|{str(default[i])}) for i,f in enumerate(ALL) if f not in NUM}
    ranges={}; statistics={}
    for i,f in enumerate(ALL):
        if f not in NUM: continue
        v=engine.tr[:,i].astype(float); lo,hi=int(v.min()),int(v.max()); q=np.quantile(v,[.25,.5,.75])
        ranges[f]=(lo,hi)
        s=f'Training median {q[1]:,.0f}; middle 50% {q[0]:,.0f}–{q[2]:,.0f}; range {lo:,}–{hi:,}.'
        if f in ['capital-gain','capital-loss']:
            pos=v[v>0]
            s+=f' Zero in {(v==0).mean()*100:.1f}% of records.'
            if len(pos): s+=f' Among nonzero values, median {np.median(pos):,.0f} USD.'
        statistics[f]=s
    # Census weight is metadata; use a stable training-derived default for custom examples.
    default[ALL.index('fnlwgt')]=int(np.median(engine.tr[:,ALL.index('fnlwgt')].astype(float)))
    targets=dict(joint='Joint — all five base models positive',**MODEL_NAMES)
    # Restrict demo samples to rows the existing form and validators can faithfully represent.
    usable=np.ones(len(engine.te),dtype=bool)
    for i,f in enumerate(ALL):
        if f in NUM:
            lo,hi=ranges[f]
            v=engine.te[:,i].astype(float)
            usable &= (v>=lo)&(v<=hi)
        else:
            usable &= np.isin(engine.te[:,i].astype(str),cats[f])
    usable &= np.array([tuple(v) in engine.jobs for v in engine.te[:,JOB]],dtype=bool)
    usable &= np.array([(str(a) in edu_map and edu_map[str(a)]==b) for a,b in engine.te[:,[ED,E]]],dtype=bool)
    rng=np.random.default_rng()
    # Previously verified failures are a fast shortlist; still recheck against loaded models.
    exact_fail={}
    exact_path=root/'exact'/'per_query.csv'
    if exact_path.exists():
        exact=pd.read_csv(exact_path,usecols=['query_id','source_model','found'])
        for key,g in exact[exact.found.eq(0)].groupby('source_model'):
            exact_fail[key]=g.query_id.to_numpy(dtype=int)
    def pick_example(action,rules,target):
        positive=(scores0[:,:5]>=0.5).all(axis=1) if target=='joint' else scores0[:,LABELS.index(target)]>=0.5
        pool=np.flatnonzero(usable & (positive if action=='pick_pass' else ~positive))
        if not len(pool): raise ValueError('No compatible held-out examples for this objective')
        if action=='pick_pass':
            order=rng.permutation(pool)[:24]
        else:
            known=np.intersect1d(pool,exact_fail.get(target,np.empty(0,dtype=int)))
            # A full-policy failure is also a failure for any subset of its actions.
            first=rng.permutation(known)[:8]
            rest=rng.permutation(np.setdiff1d(pool,first))[:24-len(first)]
            order=np.concatenate([first,rest])
        for row_id in order:
            profile=engine.te[int(row_id)].copy()
            outcome=live_options(engine,profile,rules,target)
            before,status,result,count=outcome
            already=(before[:5]>=0.5).all() if target=='joint' else before[LABELS.index(target)]>=0.5
            if (action=='pick_pass' and already) or (action=='pick_fail' and not already and result is None):
                return profile,int(row_id),outcome
        raise ValueError('No matching example verified within 24 randomly sampled profiles. Try again or choose another objective.')
    @app.route('/report/')
    def report(): return send_from_directory(root/'report','index.html')
    @app.route('/report/<path:name>')
    def asset(name): return send_from_directory(root/'report',name)
    @app.route('/',methods=['GET','POST'])
    def index():
        x=default.copy(); error=None; before=None; result=None; status=None; count=0; sample_id=None
        rules=set(request.form.getlist('rules')) if request.method=='POST' else set(DEFAULT_RULES)
        target=request.form.get('target','joint')
        try:
            if target not in targets or not rules<=set(LIVE_RULES): raise ValueError('Invalid policy or target')
            action=request.form.get('action','analyze')
            if action not in ['analyze','pick_pass','pick_fail']: raise ValueError('Invalid action')
            if request.method=='POST' and action!='analyze':
                x,sample_id,(before,status,result,count)=pick_example(action,rules,target)
            elif request.method=='POST':
                for i,f in enumerate(ALL):
                    if f=='education-num': continue
                    raw=request.form[f]
                    if f in NUM:
                        value=int(raw); lo,hi=ranges[f]
                        if not lo<=value<=hi: raise ValueError(f'{FIELD_LABELS[f]} must be between {lo} and {hi}')
                        x[i]=value
                    else:
                        if raw not in cats[f]: raise ValueError(f'Unknown {FIELD_LABELS[f]}')
                        x[i]=raw
                x[E]=edu_map[str(x[ED])]
                if tuple(x[JOB]) not in engine.jobs: raise ValueError('Choose an occupation compatible with the employer type')
                before,status,result,count=live_options(engine,x,rules,target)
        except (ValueError,KeyError) as exc: error=str(exc)
        # Derived code always follows the selected education, including validation errors.
        x[E]=edu_map[str(x[ED])]
        fields={}
        for i,f in enumerate(ALL):
            if f=='education-num': continue
            d=dict(name=f,label=FIELD_LABELS[f],help=FIELD_HELP[f],stats=statistics.get(f,''))
            if f in NUM: d.update(value=int(x[i]),options=None,min=ranges[f][0],max=ranges[f][1])
            else: d.update(value=str(x[i]),options=cats[f])
            fields[f]=d
        core=['age','education','workclass','occupation','hours-per-week','marital-status','capital-gain','capital-loss']
        groups=[('Education, employment and finances',[fields[f] for f in core]),
                ('Additional census attributes (advanced)',[fields[f] for f in fields if f not in core])]
        score_rows=[] if before is None else [(MODEL_NAMES[n],float(before[j]),float(result['after'][j]) if result else None) for j,n in enumerate(LABELS)]
        submitted=[(FIELD_LABELS[f],str(x[i])) for i,f in enumerate(ALL)]
        payload=None
        outcome=outcome_title=outcome_detail=None
        if before is not None:
            already=(before[:5]>=0.5).all() if target=='joint' else before[LABELS.index(target)]>=0.5
            if already:
                outcome='green'; outcome_title='Already positive'
                outcome_detail='The submitted profile already satisfies the selected prediction objective. No counterfactual changes are required.'
            elif result is not None:
                outcome='yellow'; outcome_title='Possible with counterfactual changes'
                outcome_detail='At least one candidate satisfies the active policies and prediction objective. Compare available effort levels below; practical availability of changes is not guaranteed.'
            else:
                outcome='red'; outcome_title='No solution under current policies'
                outcome_detail='No candidate in the defined finite action space satisfies the selected objective. This does not imply real-world impossibility.'
            payload=dict(dataset='UCI Adult historical benchmark',profile={f:str(x[i]) for i,f in enumerate(ALL)},threshold=.5,
                target=target,active_controls=sorted(rules),status=status,outcome=outcome,held_out_example_id=sample_id,permitted_candidates=count,
                original_scores=dict(zip(LABELS,before.tolist())),counterfactual=None if not result else
                dict(scores=dict(zip(LABELS,result['after'].tolist())),cost=result['cost'],changes=result['changes'],notes=result['notes']),
                valid_candidates=0 if result is None else result['valid_count'],
                distinct_cost_levels=0 if result is None else result['cost_levels'],
                counterfactual_options=[] if result is None else [
                    dict(tier=o['tier'],cost=o['cost'],edits=o['edits'],scores=dict(zip(LABELS,o['after'].tolist())),
                        changes=o['changes'],notes=o['notes'],caution=o['caution'],target_score=o['target_score'],
                        positive_base_models=o['positive_count']) for o in result['alternatives']],
                warning='Model scores only; no verified causal or real-world feasibility claim.')
        return render_template_string(PAGE,groups=groups,edu_map=edu_map,job_map=job_map,education_code=int(x[E]),missing=MISSING,
            live_rules=LIVE_RULES,rules=rules,target=target,targets=targets,before=before,result=result,error=error,status=status,
            score_rows=score_rows,submitted=submitted,candidate_count=count,positive_count=0 if before is None else int((before[:5]>=.5).sum()),
            payload=payload,report=(root/'report'/'index.html').exists(),sample_id=sample_id,
            outcome=outcome,outcome_title=outcome_title,outcome_detail=outcome_detail,
            after_count=0 if result is None else int((result['after'][:5]>=0.5).sum()))
    return app

def serve(engine,root,port):
    create_app(engine,root).run(host='127.0.0.1',port=port,debug=False)

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
