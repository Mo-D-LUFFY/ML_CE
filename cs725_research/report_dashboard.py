"""Build an offline, contextual research dashboard and project workflow from saved results."""
import argparse
import html
import json
from pathlib import Path
import pandas as pd

ROOT=Path(__file__).resolve().parent
MODELS={'lr':'Logistic Regression','rf':'Random Forest','xgb':'XGBoost','mlp':'Deep MLP','ft':'FT-Transformer','ensemble':'Soft-voting ensemble','joint':'Joint five-model objective'}
def esc(x): return html.escape(str(x),quote=True)
def pct(x): return f'{100*float(x):.2f}%'
def read(path): return pd.read_csv(path)
def meta(path): return json.loads(path.read_text(encoding='utf8'))
def table(headers,rows):
    return '<div class="table-wrap"><table><thead><tr>'+''.join('<th>'+esc(x)+'</th>' for x in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+esc(x)+'</td>' for x in row)+'</tr>' for row in rows)+'</tbody></table></div>'
def section(key,title,lead,body):
    return f'<section id="{key}"><h2>{esc(title)}</h2><p class="lead">{esc(lead)}</p>{body}</section>'
def stats(items):
    return '<div class="stats">'+''.join(f'<div class="stat"><small>{esc(k)}</small><strong>{esc(v)}</strong><span>{esc(d)}</span></div>' for k,v,d in items)+'</div>'
def note(title,body):
    return f'<div class="note"><strong>{esc(title)}</strong><p>{esc(body)}</p></div>'
def figure(out,name,title,question,how,meaning):
    if not (out/(name+'.png')).exists(): return note(title,'Figure not available in the saved report. Run research_figures.py to regenerate available plots.')
    return f'<article class="figure"><div class="figure-head"><h3>{esc(title)}</h3><a href="{name}.png" target="_blank" rel="noopener">Open full figure ↗</a></div><p>{esc(question)}</p><img loading="lazy" src="{name}.png" alt="{esc(title)}"><div class="interpret"><div><h4>How to read it</h4><p>{esc(how)}</p></div><div><h4>What it tells us</h4><p>{esc(meaning)}</p></div></div></article>'
CSS="""*{box-sizing:border-box}html{scroll-behavior:smooth;scroll-padding-top:22px}body{margin:0;font:15px/1.65 system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;background:#fff;color:#242424}a{color:inherit;text-decoration:none}a:hover{text-decoration:underline}button{font:inherit;cursor:pointer}aside{position:fixed;left:0;top:0;bottom:0;width:258px;background:#f7f7f5;border-right:1px solid #e9e9e6;padding:22px 16px;overflow:auto;z-index:5}.brand{font-size:17px;font-weight:700;letter-spacing:-.4px;display:flex;align-items:center;gap:9px;margin:0 8px 25px}.mark{display:inline-grid;place-items:center;width:28px;height:28px;border:1px solid #d6d6d1;border-radius:9px;background:#fff;font-weight:800}.navlabel{font-size:11px;text-transform:uppercase;letter-spacing:.9px;color:#73736f;font-weight:650;margin:24px 10px 8px}.navlink{display:block;border-radius:7px;padding:9px 11px;margin:3px 0;font-size:14px;color:#444}.navlink:hover,.navlink.active{background:#e9e9e6;text-decoration:none;color:#181818}.navlink.active{font-weight:650}.navlink.sub{font-size:13px;padding:7px 12px}.sidebar-foot{font-size:12px;color:#777;padding:20px 10px}.layout{margin-left:258px}.topbar{display:flex;justify-content:space-between;align-items:center;padding:16px 28px;border-bottom:1px solid #eee;color:#666;font-size:13px}.topbar a{border:1px solid #ddd;border-radius:8px;padding:5px 10px;color:#333}.menu{display:none;background:none;border:1px solid #ddd;border-radius:7px;padding:4px 9px}.content{max-width:1100px;padding:44px 42px 80px;margin:auto}.eyebrow{color:#777;font-size:12px;font-weight:650;letter-spacing:1.1px;text-transform:uppercase}.hero h1{font-size:clamp(30px,4vw,45px);line-height:1.15;letter-spacing:-1.6px;margin:10px 0 15px}.hero p{font-size:17px;max-width:800px;color:#555}.hero-links{display:flex;gap:10px;flex-wrap:wrap;margin:20px 0 10px}.button{border:1px solid #dededb;border-radius:8px;padding:9px 15px;display:inline-block;font-weight:600}.button.primary{background:#252525;color:#fff;border-color:#252525}.stats{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:12px;margin:24px 0}.stat{border:1px solid #e5e5e2;border-radius:11px;padding:16px;background:#fcfcfb;display:flex;flex-direction:column}.stat small{font-size:12px;color:#666}.stat strong{font-size:28px;letter-spacing:-.8px;line-height:1.4}.stat span{font-size:12px;color:#777}section{padding-top:30px;margin-top:30px;border-top:1px solid #e8e8e6}h2{font-size:26px;line-height:1.3;letter-spacing:-.7px;margin:0 0 10px}h3{font-size:18px;line-height:1.35;margin:0 0 7px}h4{font-size:13px;margin:0 0 6px}.lead{color:#60605e;margin:0 0 22px}.note{background:#f6f6f4;border:1px solid #e6e6e2;border-radius:9px;padding:15px 17px;margin:17px 0}.note p{margin:4px 0 0;color:#52524e}.figure-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:18px}.figure{border:1px solid #e2e2df;border-radius:12px;padding:18px;margin:0 0 16px;min-width:0;break-inside:avoid}.figure-grid .figure{margin:0}.figure-head{display:flex;justify-content:space-between;gap:10px;align-items:flex-start}.figure-head a{font-size:11px;color:#666;white-space:nowrap}.figure>p{color:#5d5d5b;font-size:14px;margin:5px 0 14px}.figure img{width:100%;height:auto;display:block;margin:12px auto;max-height:560px;object-fit:contain}.interpret{border-top:1px solid #e7e7e4;padding-top:12px;display:grid;gap:10px}.interpret p{margin:0;color:#565653;font-size:13px}.table-wrap{overflow:auto;margin:14px 0}table{width:100%;border-collapse:collapse;font-size:13px;text-align:left}th,td{padding:10px 12px;border-bottom:1px solid #e8e8e5;vertical-align:top}th{font-size:12px;color:#555;background:#f6f6f4}td:first-child{font-weight:550}.case-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:16px}.case{border:1px solid #e2e2df;border-radius:10px;padding:16px;min-width:0}.case p{font-size:13px;color:#555}.step{display:grid;grid-template-columns:38px minmax(0,1fr);gap:14px;padding:19px 0;border-bottom:1px solid #eee}.num{height:34px;width:34px;border-radius:50%;background:#f1f1ee;display:grid;place-items:center;font-weight:700}.step p{margin:6px 0;color:#555}.step code,code{font:12px ui-monospace,Consolas,monospace;background:#f2f2ef;padding:3px 5px;border-radius:4px;overflow-wrap:anywhere}.step .io{display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-top:10px}.io>div{border:1px solid #e5e5e2;border-radius:7px;padding:10px;font-size:13px}.io small{display:block;color:#777}.flow{display:grid;gap:8px;max-width:650px;margin:22px auto}.flow div{border:1px solid #ddd;border-radius:9px;background:#fafaf9;text-align:center;padding:11px;font-weight:600}.flow span{text-align:center;color:#777;line-height:1}.foot{font-size:12px;color:#777;margin-top:40px}.muted{color:#777;font-size:13px}.links{display:flex;flex-wrap:wrap;gap:12px}.links a{border-bottom:1px solid #aaa}.pill{display:inline-block;border:1px solid #ddd;border-radius:20px;padding:3px 9px;font-size:12px;color:#666}.warning{border-left:3px solid #b68b47;padding-left:12px;color:#725525}.status{font-size:12px;font-weight:650;color:#246a51}.figure-grid .figure img{max-height:360px}@media(min-width:1450px){.content{padding-left:55px;padding-right:55px}}@media(max-width:900px){aside{width:218px}.layout{margin-left:218px}.content{padding:28px 24px}.figure-grid,.case-grid{grid-template-columns:1fr}}@media(max-width:650px){aside{display:none;width:250px;box-shadow:8px 0 30px #0002}body.open aside{display:block}.layout{margin-left:0}.menu{display:inline-block}.topbar{padding:12px 16px}.content{padding:25px 16px}.hero h1{font-size:32px}.stats{grid-template-columns:1fr 1fr}.step .io{grid-template-columns:1fr}.figure-head{flex-wrap:wrap}}@media print{aside,.topbar,.hero-links,.figure-head a{display:none}.layout{margin:0}.content{padding:0}.figure-grid{grid-template-columns:1fr}.figure img{max-height:400px}section{break-before:auto}.stat,.figure,.case{break-inside:avoid}}"""
# Shared top navigation, with no fixed sidebar or duplicate section navigation.
CSS += """
.layout{margin:0}.content{max-width:1240px;padding:40px 40px 76px}.topbar{position:sticky;top:0;z-index:10;background:#fffffff5;backdrop-filter:blur(12px);display:flex;align-items:center;justify-content:space-between;gap:16px;padding:13px max(22px,calc((100% - 1240px)/2 + 40px));border-bottom:1px solid #e7e7e4}.top-brand{font-weight:750;color:#20242a;display:flex;align-items:center;gap:9px;white-space:nowrap}.topnav{display:flex;align-items:center;gap:6px;flex-wrap:wrap}.topnav a{border:0;padding:8px 13px;border-radius:8px;color:#555;font-weight:550;white-space:nowrap}.topnav a.active{background:#222;color:white}.topnav a:not(.active):hover{background:#f0f0ed;text-decoration:none}.hero{padding:12px 0 8px}.hero h1{max-width:940px}.diagram{margin:20px 0 26px;border-radius:15px;overflow:hidden;background:#111827;color:#f3f6fc;box-shadow:0 8px 28px #11182718;break-inside:avoid}.diagram-head{padding:22px 25px 8px}.diagram-head .eyebrow{color:#67d8f7}.diagram-head h3{font-size:24px;color:#f5f7fb;margin:5px 0 8px;letter-spacing:-.5px}.diagram-head p{font-size:14px;color:#c1cddd;margin:5px 0}.diagram-line{width:65px;height:3px;background:#38bdf8;margin:14px 0 0}.diagram-scroll{overflow:auto;padding:4px 17px 0}.diagram svg{display:block;width:100%;height:auto;min-width:760px}.diagram figcaption{border-top:1px solid #344155;margin:8px 23px 0;padding:14px 0 17px;color:#b8c5d7;font-size:13px}.diagram-key{display:flex;gap:16px;flex-wrap:wrap;margin:0 24px 12px;color:#b8c5d7;font-size:12px}.diagram-key span:before{content:'';display:inline-block;width:9px;height:9px;border:1px solid #73859e;margin-right:6px;background:#1c293b}.diagram-key span:nth-child(2):before{border-color:#38bdf8;background:#123044}.diagram-key span:nth-child(3):before{border-color:#35b89a;background:#13382f}.step{padding:20px 0}.step .num{background:#eaf1f7;color:#254b6c}.step .io>div{background:#fafbfc}.figure-grid{gap:20px}.figure-grid .figure img{max-height:420px}@media(max-width:900px){.layout{margin:0}.content{padding:28px 22px}.topbar{padding:12px 22px}.figure-grid,.case-grid{grid-template-columns:repeat(2,minmax(0,1fr))}}@media(max-width:700px){.topbar{position:static;align-items:flex-start;flex-direction:column;gap:8px;padding:12px 16px}.topnav{gap:3px}.topnav a{padding:7px 8px;font-size:12px}.content{padding:22px 16px}.figure-grid,.case-grid{grid-template-columns:1fr}.diagram-head{padding:17px 18px 5px}.diagram-head h3{font-size:21px}.diagram-scroll{padding:0 10px}.diagram figcaption{margin:8px 17px 0}.diagram-key{margin:0 18px 12px}}@media print{.topbar{display:none}.layout{margin:0}.content{padding:0}.diagram{print-color-adjust:exact;-webkit-print-color-adjust:exact;box-shadow:none}.diagram svg{min-width:0}.diagram-scroll{overflow:visible}}
"""
def page(title,active,nav,body):
    def link(href,label,key):
        selected=' class="active" aria-current="page"' if active==key else ''
        return f'<a href="{href}"{selected}>{esc(label)}</a>'
    return f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>{esc(title)} · Research Lab</title><style>{CSS}</style></head><body><div class="layout"><header class="topbar"><a class="top-brand" href="../"><span class="mark">R</span> Research Lab</a><nav class="topnav" aria-label="Project pages">{link('../','Live profile analysis','live')}{link('index.html','Experimental results','results')}{link('workflow.html','Project workflow','workflow')}</nav></header><main class="content">{body}<footer class="foot">CS725 · Foundations of Machine Learning · Research demonstration only. Model outputs are not causal or guaranteed real-world outcomes.</footer></main></div></body></html>'''

def diagram(key,title,lead,height,nodes,edges,caption):
    """Responsive, self-contained SVG. Coordinates describe the actual component topology."""
    colors={'default':('#192437','#71829a'),'accent':('#132d42','#38bdf8'),'result':('#14352f','#39bda1')}
    paths=''.join(f'<path d="{esc(d)}" fill="none" stroke="{color}" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" '+(f'marker-end="url(#{key}-arrow)"' if arrow else '')+'/>' for d,color,arrow in edges)
    boxes=[]
    for x,y,w,h,heading,lines,kind in nodes:
        fill,stroke=colors[kind]
        boxes.append(f'<g><rect x="{x}" y="{y}" width="{w}" height="{h}" rx="5" fill="{fill}" stroke="{stroke}" stroke-width="1.2"/><text x="{x+15}" y="{y+27}" font-size="16" font-weight="700" fill="#f3f6fc">{esc(heading)}</text>')
        for i,line in enumerate(lines):
            boxes.append(f'<text x="{x+15}" y="{y+50+17*i}" font-size="12.5" fill="#c3cfde">{esc(line)}</text>')
        boxes.append('</g>')
    return f'''<figure class="diagram"><div class="diagram-head"><div class="eyebrow">Architecture / {esc(key)}</div><h3>{esc(title)}</h3><p>{esc(lead)}</p><div class="diagram-line"></div></div><div class="diagram-scroll"><svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1000 {height}" role="img" aria-label="{esc(title)}" style="font-family:system-ui,sans-serif"><title>{esc(title)}</title><desc>{esc(caption)}</desc><defs><marker id="{key}-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M0 0L10 5L0 10Z" fill="#38bdf8"/></marker></defs>{paths}{''.join(boxes)}</svg></div><div class="diagram-key"><span>Input / component</span><span>Processing / decision</span><span>Saved or returned output</span></div><figcaption>{esc(caption)}</figcaption></figure>'''

def architecture_diagrams(m):
    cyan='#38bdf8'; muted='#71829a'
    def edge(d,arrow=True,color=cyan): return (d,color,arrow)
    training=diagram('01','Training and model architecture','One official dataset, two feature configurations, five separately trained classifiers.',530,[
        (325,12,350,65,'UCI Adult — official split',[f'{m["n_train"]:,} training / {m["n_test"]:,} test records'],'default'),
        (45,116,420,82,'Training and validation',['Fit preprocessing on training data','Select neural epochs; refit on all training rows'],'accent'),
        (705,116,250,82,'Held-out test set',['Never used to fit model parameters','Evaluate after final refit'],'default'),
        (30,247,170,65,'Logistic Regression',['Linear + sigmoid'],'default'),
        (222,247,170,65,'Random Forest',['120 trees'],'default'),
        (414,247,170,65,'XGBoost',['Gradient boosting'],'default'),
        (606,247,170,65,'Deep MLP',['128 → 64 → 32'],'default'),
        (798,247,172,65,'FT-Transformer',['Feature-token attention'],'default'),
        (45,355,420,65,'Saved model checkpoints',['Fitted preprocessors + five classifiers'],'result'),
        (540,355,415,65,'Test predictions and metrics',['Accuracy, F1, ROC-AUC; aligned scores'],'result'),
        (540,455,415,65,'Soft-voting ensemble',['Mean of five positive-class scores'],'accent'),
    ],[
        edge('M500 77V96H255V116'),edge('M500 96H830V116'),
        edge('M255 198V224H115',False),edge('M255 224H884',False),
        *[edge(f'M{x} 224V247') for x in [115,307,499,691,884]],
        *[edge(f'M{x} 312V333',False) for x in [115,307,499,691,884]],
        edge('M115 333H884',False),edge('M255 333V355'),
        edge('M748 333V355'),edge('M955 157H984V385H955'),
        edge('M748 420V455')
    ],'Training and test data are separated. The five model checkpoints feed the research and live inference systems. Soft voting combines their outputs, not their training data.')
    search=diagram('02','Counterfactual search and policy validation','A common-negative query enters multiple search strategies; model scores and independent rules determine valid choices.',522,[
        (325,12,350,65,'Original held-out profile',['All five base predictions initially negative'],'default'),
        (45,118,420,82,'Baseline candidate generation',['Nearest-positive / unconstrained donor search','No actionability restriction during generation'],'accent'),
        (535,118,420,82,'Policy-aware candidate generation',['Constrained random / exact hours / full enumeration','Hours, education and employment edits'],'accent'),
        (45,247,420,82,'Model inference',['Evaluate the same five trained classifiers','Store each positive-class score'],'accent'),
        (535,247,420,82,'Independent policy audit',['Frozen features, coherent categories, bounds','Compute cost, distance and caution flags'],'accent'),
        (45,378,280,82,'Baseline selection',['Source flip + nearest distance','Record violations separately'],'default'),
        (360,378,280,82,'Constrained selection',['Hard-feasible + source positive','Minimize cost, then distance'],'result'),
        (675,378,280,82,'Joint robust selection',['Hard-feasible + all five positive','Minimize cost, then distance'],'result'),
    ],[
        edge('M500 77V96H255V118'),edge('M500 96H745V118'),
        edge('M255 200V224H745',False),edge('M745 200V224',False),
        edge('M255 224V247'),edge('M745 224V247'),
        edge('M255 329V348H185V378'),edge('M255 348H815',False),
        edge('M745 329V348',False),edge('M500 348V378'),edge('M815 348V378')
    ],'Baseline selections can be prediction-valid while violating the policy. Constrained and joint selections must pass both model and rule checks. Exhaustive enumeration establishes the optimum only within the defined finite action space.')
    app=diagram('03','System architecture — live inference and research reports','One local Flask application serves the custom-profile tool and the offline research documentation.',525,[
        (45,14,280,65,'User / TA',['Browser · custom input or example'],'default'),
        (375,14,580,65,'Flask application',['research_v3.py · localhost:5000'],'accent'),
        (45,133,280,82,'Profile input and controls',['Education mapping / job dropdown','Seven independent action controls'],'default'),
        (360,133,280,82,'Saved model engine',['Five fitted models + preprocessing','Six scores including soft voting'],'accent'),
        (675,133,280,82,'Saved experiment artifacts',['Metrics, query results, plots','Bootstrap intervals and case studies'],'default'),
        (45,270,280,82,'Live policy search',['Enumerate permitted changes','Evaluate candidates with model engine'],'accent'),
        (360,270,280,82,'Current profile report',['Already positive / changes / no solution','Min, middle, max valid effort levels'],'result'),
        (675,270,280,82,'Experimental dashboard',['report_dashboard.py','Context, findings and figures'],'result'),
        (360,408,280,75,'Report export',['Print / PDF or JSON','Current submitted profile'],'result'),
        (675,408,280,75,'Project workflow',['Architecture and code walkthrough','Reproduction instructions'],'result'),
    ],[
        edge('M325 46H375'),edge('M665 79V105H185V133'),edge('M665 105H815V133'),
        edge('M500 105V133'),edge('M185 215V270'),edge('M360 174H325'),
        edge('M325 311H360'),edge('M500 215V270'),edge('M185 270V239H500V215',True,muted),
        edge('M815 215V270'),edge('M500 352V408'),edge('M815 352V408')
    ],'The live page computes a fresh report using checkpoints. The experimental results and workflow pages read saved artifacts, so rebuilding them does not train or modify any classifier.')
    return training+search+app

def grid(*figs): return '<div class="figure-grid">'+''.join(figs)+'</div>'
def case_card(c,title):
    rows=[[k,v['before'],v['after']] for k,v in c['changes'].items()]
    p=[[MODELS[m],pct(c['original_scores'][m]),pct(c['candidate_scores'][m])] for m in ['lr','rf','xgb','mlp','ft']]
    detail='Policy compliant; conditional actions require verification.' if c['feasible'] else 'Policy violation: '+str(c['reason']).replace('_',' ')
    return f'<article class="case"><h3>{esc(title)}</h3><p>Held-out query {esc(c["query_id"])} · {esc(detail)}</p>{table(["Feature","Before","After"],rows)}{table(["Model","Original","Modified"],p)}<p>Caution: {esc(c["caution"])}</p></article>'

def build(root,out):
    out.mkdir(parents=True,exist_ok=True)
    m=meta(root/'counterfactual'/'metadata.json')
    n=int(m['n_queries'])
    cl=read(out/'classification.csv') if (out/'classification.csv').exists() else pd.concat([read(root/f/'metrics.csv') for f in ['classical','mlp','transformer']]).query("split=='test'")
    cf=read(root/'counterfactual'/'summary.csv')
    ex=read(root/'exact'/'summary.csv')
    if not cf.queries.eq(n).all() or not ex.queries.eq(n).all(): raise ValueError('Incomplete or inconsistent query counts; finish the experiments first.')
    full=cl[cl.features.eq('full')].copy()
    best=full.loc[full.f1.idxmax()]
    def mean(method,col): return float(cf.loc[cf.method.eq(method)&cf.source_model.isin(['lr','rf','xgb','mlp','ft']),col].mean())
    u=mean('unconstrained_random','flip_coverage'); uf=mean('unconstrained_random','feasible_coverage'); c=mean('constrained_random','feasible_coverage'); near=mean('nearest_positive','feasible_coverage')
    j=ex[ex.source_model.eq('joint')].iloc[0]; exact=float(j.exact_coverage); rand=float(j.random_coverage)
    extra=int(round(n*(exact-rand)))
    nav=[('overview','Overview'),('models','Predictive models'),('validity','Validity & feasibility'),('robustness','Cross-model agreement'),('optimization','Exact optimization'),('case','Case study'),('extensions','V3 experiments'),('limits','Limitations & files')]
    body=f'<header class="hero" id="overview"><div class="eyebrow">Experimental evidence / UCI Adult</div><h1>When is a counterfactual actually actionable?</h1><p>We compare prediction validity, policy compliance and agreement across machine-learning architectures. Read the key results first, then inspect the evidence and methodology behind each finding.</p><div class="hero-links"><a class="button primary" href="workflow.html">Understand the project workflow</a><a class="button" href="../">Try a custom profile</a></div></header>'
    body+=stats([('Real records',f'{m["n_train"]+m["n_test"]:,}','Official Adult split'),('Test queries',f'{n:,}','Initially negative across all five'),('Best base-model F1',f'{best.f1:.4f}',MODELS.get(best.model,best.model)),('Joint exact coverage',pct(exact),'All five classifiers positive')])
    body+=note('The central finding',f'Unconstrained search flips predictions for {pct(u)} of queries on average, but only {pct(uf)} receive policy-compliant candidates. Explicit constraints improve feasible coverage to {pct(c)}; exhaustive joint search finds a valid candidate for {pct(exact)} of profiles.')
    def F(name,title,q,how,meaning): return figure(out,name,title,q,how,meaning)
    b=table(['Model','Features','Accuracy','F1','ROC-AUC'],[[MODELS.get(r.model,r.model),r.features,pct(r.accuracy),f'{r.f1:.4f}',f'{r.roc_auc:.4f}'] for r in cl.itertuples()])
    b+=grid(F('classification_f1','F1 by feature set','Does using all 14 inputs improve positive-class prediction?','Compare the two bars for each model; F1 balances precision and recall.',f'{MODELS.get(best.model,best.model)} has the highest full-feature F1 ({best.f1:.4f}) among the base models tested.'),
        F('classification_roc_auc','ROC-AUC by feature set','How well can models rank positive examples above negatives?','Higher AUC means stronger ranking across thresholds, not necessarily better calibration.','Compare feature sets and architectures without relying on one threshold.'))
    b+=grid(F('classification_accuracy','Accuracy by feature set','What fraction of labels are predicted correctly?','The classes are imbalanced, so accuracy is not sufficient by itself.','Read this together with F1 and ROC-AUC.'),
        F('roc_curves','ROC curves','What happens as the classification threshold varies?','Horizontal axis: false positive rate. Vertical axis: true positive rate. The diagonal is chance ranking.','Curves closer to the top-left indicate stronger discrimination. These are official test-set scores.'))
    b+=F('training_curves','Neural training and validation loss','When do the MLP and Transformer stop improving on validation data?','The validation curves determine the selected epoch count. The final models are refitted on the full training split.','Longer training is not automatically better; validation-based stopping limits overfitting.')
    body+=section('models','01 / Predictive performance','Five architectures, two feature sets, and a shared official test split. Prediction quality is not the same as recourse availability.',b)
    b=grid(F('cf_found','Prediction-flip coverage','How often does each search method find a positive prediction?','Bars divide successful source-model flips by all queries. Nearest positive uses existing training rows; random methods use matched proposal attempts.','High flip coverage alone does not make a suggestion actionable.'),
        F('cf_feasible','Hard-feasible coverage','How often does a returned counterfactual also satisfy our policy?','The denominator is all queries, including failures. Hard feasibility is checked separately from prediction.',f'Nearest-positive feasible coverage is {pct(near)}; constrained random search reaches {pct(c)} on average.'))
    b+=note('Interpret the comparison carefully','Random search conditions match proposal attempts, not unique candidates. They differ in proposal space and ranking. Hard-feasible means the implemented rules are satisfied, not verified personal access or a causal effect.')
    body+=section('validity','02 / Validity versus feasibility','A model may respond to impossible or inappropriate changes. The validator freezes historical and demographic attributes and checks permitted action groups.',b)
    b=grid(F('transfer_unconstrained_random','Unconstrained transfer','Does a counterfactual selected for one model also flip other models?','Rows indicate the source classifier; columns indicate the evaluated classifier. Diagonal cells equal source flip coverage. All queries are counted.','Off-diagonal cells show that architectures may disagree even when the source prediction flips.'),
        F('transfer_constrained_random','Constrained transfer','Does policy-compliant recourse transfer across architectures?','Each cell is unconditional coverage, including search failures. For conditional transfer divide by source success coverage.','Joint agreement is stricter than success for any one source model.'))
    b+=note('Joint objective','We require all five base-model scores to reach 0.5 simultaneously. The ensemble is their mean, so it is automatically positive when all five are positive; it is not a sixth independent source of evidence.')
    body+=section('robustness','03 / Cross-model robustness','Different learning mechanisms produce different decision boundaries. A shared counterfactual is more stable across this fixed collection of models.',b)
    b=stats([('Joint random',pct(rand),f'{round(rand*n):,} of {n:,} queries'),('Joint exhaustive',pct(exact),f'{round(exact*n):,} of {n:,} queries'),('Additional successes',str(extra),'Recovered by enumeration')])
    b+=grid(F('exact_coverage','Random versus exhaustive search','Are random-search failures caused by incomplete exploration?','Compare methods on identical query IDs, policy and ranking. Exact search has a larger variable budget.',f'Joint coverage improves from {pct(rand)} to {pct(exact)}. The difference identifies missed solutions in the sampled search.'),
        F('paired_cost_reduction','Paired effort reductions','Can exhaustive search find cheaper candidates for queries that already succeeded?','Positive values are random cost minus exact cost, calculated only on shared successes. Box plots summarize the distribution.','Exact optimization cannot have a higher minimum cost within the same finite domain, subject to numerical evaluation.'))
    b+=table(['Objective','Random coverage','Exact coverage','Gain (pp)'],[[MODELS.get(r.source_model,r.source_model),pct(r.random_coverage),pct(r.exact_coverage),f'{100*(r.exact_coverage-r.random_coverage):+.2f}'] for r in ex.itertuples()])
    ci=out/'confidence_intervals.csv'
    if ci.exists():
        d=read(ci); d=d[d.metric.eq('extra')]
        b+='<h3>Paired bootstrap uncertainty</h3>'+table(['Objective','Gain (pp)','95% interval (pp)'],[[MODELS.get(r.model,r.model),f'{100*r.estimate:.2f}',f'{100*r.ci_low:.2f} to {100*r.ci_high:.2f}'] for r in d.itertuples()])+'<p class="muted">5,000 paired query resamples; uncertainty is conditional on the fixed models, cohort and policy, not retraining.</p>'
    body+=section('optimization','04 / Search and optimization','We enumerate permitted integer hours, coherent education pairs and observed employment pairs. A missing solution is only a statement about this defined action domain.',b)
    cases=meta(root/'counterfactual'/'cases.json')
    selected=next((x for x in cases if x['method']=='joint_robust'),None)
    blocked=next((x for x in cases if selected is not None and x['query_id']==selected['query_id'] and not x['feasible']),None)
    b='<div class="case-grid">'+(case_card(blocked,'Prediction valid, policy blocked') if blocked else '')+(case_card(selected,'Joint, policy-compliant candidate') if selected else '')+'</div>'
    if not selected: b=note('No case available','No joint success exists in the saved cases.')
    b+=note('Why this example matters','A valid model score is not a verified intervention. Compare the modified columns and the response of each classifier. Employment and education changes can be conditional even when the hard rules pass.')
    body+=section('case','05 / Trace a real held-out profile','Two outcomes from the same saved query when available, illustrating the distinction between a prediction flip and a permitted counterfactual.',b)
    b=''
    for tag in ['six','full']:
        p=root/'v3'/f'ensemble_{tag}_metrics.csv'
        if p.exists():
            d=read(p)
            b+=f'<h3>Adult ensemble · {tag} features</h3>'+table(['Model','Accuracy','F1','ROC-AUC'],[[MODELS.get(r.model,r.model),pct(r.accuracy),f'{r.f1:.4f}',f'{r.roc_auc:.4f}'] for r in d.itertuples()])
    p=root/'v3'/'policy_summary.csv'
    if p.exists():
        d=read(p); d=d[d.target.isin(['joint','ensemble'])]
        b+='<h3>Policy sensitivity · saved run</h3>'+table(['Policy','Target','Queries','Coverage','Median cost'],[[r.policy,MODELS.get(r.target,r.target),r.queries,pct(r.coverage),'—' if pd.isna(r.median_cost) else f'{r.median_cost:.1f}'] for r in d.itertuples()])
    for ds in ['acs','plfs']:
        p=root/'modern'/ds/'metrics.csv'; mp=p.parent/'metadata.json'
        if p.exists() and mp.exists():
            d=read(p); mm=meta(mp)
            b+=f'<h3>{esc(mm["dataset"])}</h3><p>{esc(mm.get("target",""))} · {esc(mm.get("n_eligible","unknown"))} eligible records · saved classifier rows: {len(d)}. A partial run is not a completed benchmark.</p>'+table(['Model','Accuracy','F1','ROC-AUC'],[[MODELS.get(r.model,r.model),pct(r.accuracy),f'{r.f1:.4f}',f'{r.roc_auc:.4f}'] for r in d.itertuples()])
            fp=p.parent/'cf_summary.csv'
            if fp.exists():
                dd=read(fp); dd=dd[dd.target.isin(['joint','ensemble'])]
                b+=table(['Search','Target','Queries','Coverage','Feasible coverage'],[[r.method,r.target,r.queries,pct(r.coverage),pct(r.feasible_coverage)] for r in dd.itertuples()])
        else: b+=note(ds.upper()+' benchmark','No saved classification results found; not included in measured findings.')
    if not (root/'v3'/'ensemble_full_metrics.csv').exists(): b+=note('Adult ensemble evaluation','The live predictor is implemented, but saved ensemble metrics are not present in this results directory.')
    body+=section('extensions','06 / Research V3 extensions','Only saved local results appear here. ACS and PLFS use different targets, cohorts and feature meanings; numerical performance or effort is not directly comparable across countries.',b)
    b=note('Scope of evidence','Adult uses historical census records. The study uses fixed models, a common-negative cohort, illustrative action costs and rule-defined feasibility. No causal income guarantee, fairness audit, independent retraining robustness or DiCE benchmark is established.')
    b+='<div class="links">'+''.join(f'<a href="{f}">{label} ↓</a>' for f,label in [('classification.csv','Classification CSV'),('confidence_intervals.csv','Bootstrap intervals'),('paired_costs.csv','Paired effort costs')] if (out/f).exists())+'</div>'
    body+=section('limits','07 / Limitations & reproducibility','Download the numerical results, inspect the figures, or follow the workflow to understand how every stage is produced.',b)
    (out/'index.html').write_text(page('Experimental results','results',nav,body),encoding='utf8')
    workflow(out,m)
    print(f'Dashboard and workflow: {out.resolve()}')

def workflow(out,m):
    nav=[('purpose','Research question'),('pipeline','End-to-end pipeline'),('training','Data & models'),('search','Counterfactual search'),('analysis','Evaluation & reporting'),('run','Reproduce the project')]
    body='<header class="hero"><div class="eyebrow">Project guide / start here</div><h1>From census data to actionable explanations</h1><p>A walkthrough for someone seeing the project for the first time. Follow the data, model checkpoints, action rules and evaluation outputs through the actual implementation.</p><div class="hero-links"><a class="button primary" href="index.html">Explore experimental results</a><a class="button" href="../">Try live inference</a></div></header>'
    body+=section('purpose','01 / What problem are we solving?','An income classifier predicts a binary label. A counterfactual asks what input changes would flip that prediction. Our research asks whether those changes are permitted and whether multiple models agree.',note('Illustrative example','A score of 0.32 becomes 0.55 after a hypothetical employment change. That is prediction validity. A separate rule check determines policy compliance. Real-world availability remains unverified.'))
    body+=section('pipeline','02 / Architecture diagrams','Three views of the same implemented system: predictive training, counterfactual generation, and the live application with its research outputs.',architecture_diagrams(m))
    def step(i,title,desc,file,inp,outp):
        return f'<article class="step"><div class="num">{i}</div><div><h3>{esc(title)}</h3><p>{esc(desc)}</p><code>{esc(file)}</code><div class="io"><div><small>Input</small>{esc(inp)}</div><div><small>Output</small>{esc(outp)}</div></div></div></article>'
    b=step(1,'Load and clean','Read the official Adult split; normalize labels, recognize missing data and record file hashes. The initial validation split comes from training records only.','train_classical.py',f'{m["n_train"]:,} training + {m["n_test"]:,} test records','Clean features, target and train/validation split')
    b+=step(2,'Train classical classifiers','Logistic Regression, Random Forest and XGBoost use fitted numerical imputation/scaling and categorical one-hot encoding. Final fitting uses all official training records.','train_classical.py','6-feature and 14-feature configurations','Saved joblib pipelines, metrics, test scores')
    b+=step(3,'Train neural classifiers','The MLP learns nonlinear hidden representations. The FT-style Transformer learns 48-dimensional feature tokens and attention interactions. AdamW and validation BCE choose training duration before full refit.','train_mlp.py / train_transformer.py','Same official split and feature sets','PyTorch checkpoints, preprocessors, loss histories')
    b+=step(4,'Combine predictions','Equal-weight soft voting averages the five positive-class scores. The ensemble is evaluated on saved test predictions without learning weights from the test set.','research_v3.py ensemble','Five aligned test-score columns','Ensemble predictions and metrics')
    body+=section('training','03 / Data and predictive models','All models estimate the positive class at a threshold of 0.5; they are the experimental environment, not the entire research contribution.',b)
    b=step(5,'Choose query cohort','Use the five full-feature models to select records initially negative across all five. Sample common IDs for matched comparisons.','run_counterfactuals.py',f'{m.get("eligible_common_negative","Common-negative")} eligible records',f'{m["n_queries"]:,} shared queries in this run')
    b+=step(6,'Generate candidates','Nearest positive searches observed training rows; donor-based unconstrained and constrained search use equal proposal attempts. Strict search enumerates allowed hours.','run_counterfactuals.py','Original profile, training donors, proposal budget','Candidate profiles and five-model score arrays')
    b+=step(7,'Validate and rank','Freeze all but hours, education pair and employment pair. Check bounds and observed category pairs. Rank valid constrained candidates by cost, then distance. Cost = extra hours + 10 × education-level increase + 20 × employment-pair change.','run_counterfactuals.py / assess() / rank()','Candidates, original profile, policy','Feasibility flag, caution, cost and chosen candidate')
    b+=step(8,'Require model agreement','Joint selection filters the same constrained pool to candidates positive across all five models. Exhaustive search enumerates the complete finite product of permitted hours, education pairs and employment pairs.','run_exact.py','Same held-out queries and action policy','Exact coverage, minimum cost and search-gap comparison')
    b+=note('Live alternatives','The live website ranks the permitted candidates at the minimum, middle distinct, and maximum valid cost levels. The higher-cost choices are comparisons, not recommendations. User controls restrict the same action space.')
    body+=section('search','04 / Counterfactual generation','No model is retrained during the search. Inference is batched, and a rule-based validator is independent of the classifier.',b)
    b=step(9,'Measure outcomes','Classification metrics include accuracy, F1, ROC-AUC and recall. Counterfactual coverage uses all query IDs, including failures. Transfer matrices compare source versus evaluated model.','research_figures.py','Saved model and per-query result CSVs','Figures, paired bootstrap intervals, paired costs')
    b+=step(10,'Explore the results','The experimental dashboard summarizes offline research; the live profile page generates a fresh report for a submitted hypothetical record.','report_dashboard.py / research_v3.py demo','Saved figures and trained checkpoints','Contextual results, workflow and individual reports')
    b+=step(11,'Extend to modern data','ACS and PLFS adapters define dataset-specific features and earnings labels. Results appear only if their local metric files exist; a partial run is explicitly labeled.','train_modern.py','New survey data and grouped splits','Separate checkpoints, metrics and counterfactual outcomes')
    body+=section('analysis','05 / Evaluation and reporting','Separate measured benchmark results from individual hypothetical examples and unfinished experiments.',b)
    cmds=['python cs725_research/train_classical.py','python cs725_research/train_mlp.py','python cs725_research/train_transformer.py','python cs725_research/run_counterfactuals.py --queries 1000 --budget 512','python cs725_research/run_exact.py --queries 1000','python cs725_research/research_v3.py ensemble','python cs725_research/research_v3.py policies --queries 1000','python cs725_research/research_figures.py','python cs725_research/report_dashboard.py','python cs725_research/research_v3.py demo']
    body+=section('run','06 / Reproduce the project','Run from the repository root in the existing virtual environment. If the experiments already exist, only the dashboard command and demo are needed to refresh the presentation.', ''.join(f'<p><code>{esc(c)}</code></p>' for c in cmds)+note('Do not confuse the targets','Adult uses the historical US income label. ACS uses modern US annual personal income. PLFS uses Indian salaried earnings. Train separately; only verified completed runs should enter the final report.'))
    (out/'workflow.html').write_text(page('Project workflow','workflow',nav,body),encoding='utf8')

def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--results',type=Path,default=ROOT/'results'); p.add_argument('--out',type=Path); a=p.parse_args()
    build(a.results,a.out or a.results/'report')
if __name__=='__main__': main()
