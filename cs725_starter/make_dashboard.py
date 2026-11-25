"""Create a self-contained offline HTML TA demo from run_demo.py artifacts."""
import json
from pathlib import Path

ROOT = Path(__file__).parent

def build_dashboard():
    out = ROOT / 'outputs'
    d = {
        'metadata': json.loads((out / 'run_metadata.json').read_text()),
        'metrics': json.loads((out / 'classifier_metrics.json').read_text()),
        'summary': json.loads((out / 'comparison_summary.json').read_text()),
        'cases': json.loads((out / 'query_examples.json').read_text()),
        'example': json.loads((out / 'demo_example.json').read_text())
    }
    embedded = json.dumps(d, ensure_ascii=True).replace('<', '\\u003c')
    html = r'''<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>CS725 — Counterfactual Feasibility | Demo</title>
<style>
:root{--bg:#f6f7f9;--ink:#1d2735;--soft:#637187;--line:#dee3e9;--navy:#172a47;--accent:#165f8a;--white:white}
*{box-sizing:border-box} body{background:var(--bg);color:var(--ink);margin:0;font:15px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",Arial,sans-serif}
header{background:var(--navy);color:white;padding:27px max(25px,calc((100vw - 1180px)/2)) 26px} .eyebrow{text-transform:uppercase;letter-spacing:1.5px;font-size:11px;opacity:.8;font-weight:700}
h1{font-size:28px;letter-spacing:-.6px;line-height:1.2;margin:7px 0 10px} .sub{color:#d3deec;max-width:870px;margin:0}
main{max-width:1180px;margin:auto;padding:24px 24px 55px}.alert{border:1px solid #e6cb88;background:#fffaf0;border-radius:9px;padding:13px 17px;color:#75561c;margin-bottom:20px}
section{margin:23px 0}h2{font-size:19px;letter-spacing:-.25px;margin:0 0 13px}h3{font-size:15px;margin:0 0 10px}
.cards{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}.metric,.panel{border:1px solid var(--line);background:white;border-radius:10px;padding:17px;min-width:0}.metric strong{display:block;font-size:29px;color:var(--navy);line-height:1.2;margin-top:5px}.label{font-size:12px;font-weight:650;color:var(--soft)}.micro{font-size:12px;color:var(--soft)}
.controls{display:flex;gap:12px;flex-wrap:wrap;align-items:end}label{display:block;font-size:12px;font-weight:700;color:var(--soft);margin-bottom:5px}select{background:white;max-width:100%;border:1px solid #cbd4df;padding:10px 12px;border-radius:7px;font:inherit;min-width:220px}
.profile-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:9px}.kv{border:1px solid var(--line);border-radius:7px;padding:9px 11px}.kv .v{font-weight:650;margin-top:3px;font-size:16px;word-wrap:break-word}
.twocol{display:grid;grid-template-columns:repeat(2,1fr);gap:14px}.panel small{color:var(--soft)}
.pill{display:inline-block;padding:2px 8px;border-radius:15px;font-size:12px;font-weight:800;letter-spacing:.2px}.blocked{background:#f9dede;color:#822424}.caution{background:#faeacb;color:#745011}.pass{background:#dcefe6;color:#236143}.no-result{background:#e9edf2;color:#536172}
.change{border:1px solid var(--line);border-radius:6px;margin:7px 0;padding:10px 11px}.change b{color:var(--navy)}.empty{color:var(--soft);padding:28px 0;font-style:italic}
#summaryTable{border-collapse:collapse;width:100%;min-width:580px}th,td{border-bottom:1px solid var(--line);padding:10px 11px;text-align:left;font-size:13px}th{background:#f0f3f7;color:#344357}table tr:hover td{background:#fafcfd}.scroll{overflow:auto}
footer{border-top:1px solid var(--line);margin-top:27px;padding-top:16px;color:var(--soft);font-size:12px} .prob{font-size:22px;font-weight:750;color:var(--navy)}
@media(max-width:760px){.cards{grid-template-columns:repeat(2,1fr)}.twocol{grid-template-columns:1fr}.profile-grid{grid-template-columns:repeat(2,1fr)}h1{font-size:22px}}
</style></head><body>
<header><div class="eyebrow">CS 725 · Foundations of Machine Learning · Prototype v0.1</div>
<h1>When is a counterfactual actually actionable?</h1>
<p class="sub">Compare a prediction-changing suggestion with an independent warning check. A model flip alone does not mean that the proposed change can be made in the real world.</p></header>
<main><div class="alert" id="dataNotice"></div>
<section><h2>1. Held-out classifier performance</h2><div class="controls" style="margin-bottom:12px"><div><label for="modelChoice">Prediction model</label><select id="modelChoice"><option value="logistic_regression">Logistic Regression</option><option value="random_forest">Random Forest</option></select></div><div class="micro" id="modelNote"></div></div><div class="cards" id="metrics"></div></section>
<section><h2>2. Explore actual generated cases</h2><div class="controls" style="margin-bottom:14px"><div><label for="caseChoice">Held-out negative profile</label><select id="caseChoice"></select></div><div class="micro">Both methods receive the same original input; the classifier is fixed.</div></div>
<div class="panel"><h3>Original profile</h3><div class="profile-grid" id="original"></div><div style="margin-top:12px"><span class="label">Classifier prediction</span><div class="prob" id="originalProb"></div></div></div>
<div class="twocol" style="margin-top:14px"><div class="panel"><h3>A · Unconstrained counterfactual</h3><div id="unconstrained"></div></div><div class="panel"><h3>B · Constraint-aware counterfactual</h3><div id="constrained"></div></div></div></section>
<section><h2>3. Aggregate prototype comparison</h2><div class="panel scroll"><table id="summaryTable"><thead><tr><th>Classifier</th><th>Generator</th><th>Queries</th><th>Prediction flips</th><th>Feasible flips</th><th>Blocked outputs</th></tr></thead><tbody id="summary"></tbody></table></div><p class="micro">Denominators are the same sampled negative queries for both methods within each classifier. These early generators use different search spaces and budgets, so differences are descriptive, not a causal measurement of constraints alone.</p></section>
<section class="panel"><h2>How warning decisions work</h2><p><span class="pill blocked">BLOCKED</span> Changing frozen history (age, education, marital status, recorded capital gain), exceeding work-hour limits, or changing occupation without explicit career access.</p><p><span class="pill caution">CAUTION</span> A proposal obeys the hard policy, but would add more than 10 weekly hours or needs an unverified occupation change.</p><p><span class="pill pass">PASS</span> No rule in this limited policy flagged the candidate. This is not certification that it will work in the real world.</p><p class="micro">Warning system = transparent rules, not a separately trained machine-learning model. No causal claim is made about actual income.</p></section>
<footer>Source: UCI Adult, Becker &amp; Kohavi (1996), DOI 10.24432/C5XW20. This is an offline viewer built entirely from this project's generated output files. No tracking or network calls.</footer>
</main>
<script>
const DATA = __DATA__;
const Q = x => document.getElementById(x);
const F = x => Number(x).toFixed(3);
const escapeHtml = x => String(x).replaceAll('&','&amp;').replaceAll('<','&lt;').replaceAll('>','&gt;').replaceAll('"','&quot;');
const names = {'age':'Age','education-num':'Education level (code)','hours-per-week':'Work hours/week','capital-gain':'Recorded capital gain','occupation':'Occupation','marital-status':'Marital status'};
const modelNames = {logistic_regression:'Logistic Regression',random_forest:'Random Forest'};
function drawModel(){
 const m=Q('modelChoice').value, metrics=DATA.metrics[m], cases=DATA.cases[m];
 Q('modelNote').textContent = `${DATA.metadata.n_train} training rows · ${DATA.metadata.n_test} test rows · seed ${DATA.metadata.random_seed}`;
 Q('metrics').innerHTML = [['Accuracy',metrics.accuracy],['Positive F1',metrics.positive_f1],['ROC AUC',metrics.roc_auc],['Balanced accuracy',metrics.balanced_accuracy]]
 .map(([name,num])=>`<div class="metric"><span class="label">${name}</span><strong>${F(num)}</strong></div>`).join('');
 Q('caseChoice').innerHTML=cases.map((x,i)=>`<option value="${i}">Case ${i+1} — ${escapeHtml(x.query_id)}${x.constrained ? ' · feasible flip' : ''}</option>`).join('');
 const selected=cases.findIndex(e => e.query_id === DATA.example.query_id);
 Q('caseChoice').selectedIndex = selected>=0 && m==='logistic_regression' ? selected : Math.max(0,cases.findIndex(e => e.constrained));
 drawCase();
}
function drawCard(c){
 if(!c) return `<div class="empty">No prediction-flipping counterfactual found within this search budget.</div><span class="pill no-result">NO CANDIDATE</span>`;
 const status=c.verdict.toLowerCase();
 const changes=c.changes.map(x=>`<div class="change"><b>${escapeHtml(names[x.feature]||x.feature)}</b><div>${escapeHtml(x.from)} → <strong>${escapeHtml(x.to)}</strong></div></div>`).join('');
 const alerts=c.warnings.map(x=>`<div style="margin-top:8px;font-size:13px"><strong>${escapeHtml(x.code)}</strong> — ${escapeHtml(x.message)}</div>`).join('');
 return `<span class="pill ${status}">${escapeHtml(c.verdict)}</span><div style="margin:10px 0 3px" class="label">Probability of predicted &gt;$50K class</div>
 <div class="prob">${(100*c.counterfactual_probability_gt_50k).toFixed(1)}% <span class="micro">model output</span></div>
 <p class="micro">Prediction changed from &lt;=50K to &gt;50K. ${c.hard_constraints_satisfied?'Hard rules satisfied.':'Hard rules violated.'}</p>
 <div class="label">Proposed changes</div>${changes}${alerts}
 <p class="micro">Changed features: ${c.change_count} · Hypothetical cost: ${c.estimated_cost_units===null?'not applicable':c.estimated_cost_units+' units'}</p>`;
}
function drawCase(){
 const m=Q('modelChoice').value, cases=DATA.cases[m];
 const entry=cases[Number(Q('caseChoice').value)] || cases[0]; if(!entry) return;
 Q('original').innerHTML=Object.entries(entry.original).map(([key,v])=>`<div class="kv"><div class="label">${escapeHtml(names[key]||key)}</div><div class="v">${escapeHtml(v)}</div></div>`).join('');
 const p=entry.unconstrained?.original_probability_gt_50k??entry.constrained?.original_probability_gt_50k;
 Q('originalProb').textContent='≤$50K'+(p===undefined?'':` · ${(p*100).toFixed(1)}% model score for >$50K`);
 Q('unconstrained').innerHTML=drawCard(entry.unconstrained);
 Q('constrained').innerHTML=drawCard(entry.constrained);
}
function start(){
 Q('dataNotice').innerHTML = DATA.metadata.is_synthetic
 ? '<strong>DEMO DATA: SYNTHETIC.</strong> These results were computed from invented Adult-like records. They are not findings on the UCI Adult dataset. Use the original downloaded Adult files for real benchmark experiments.'
 : '<strong>UCI ADULT BENCHMARK.</strong> These model outputs use historical observational income data. Suggestions are hypothetical, not income or employment advice.';
 Q('summary').innerHTML=DATA.summary.map(r=>`<tr><td>${escapeHtml(modelNames[r.classifier])}</td><td>${escapeHtml(r.generator)}</td><td>${r.queries}</td><td>${r.flip_success_count}/${r.queries}</td><td>${r.feasible_flip_count}/${r.queries}</td><td>${r.blocked_candidate_count}</td></tr>`).join('');
 Q('modelChoice').addEventListener('change',drawModel);Q('caseChoice').addEventListener('change',drawCase);
 drawModel();
}
start();
</script></body></html>'''
    html = html.replace('__DATA__', embedded)
    target = ROOT / 'demo_dashboard.html'
    target.write_text(html, encoding='utf8')
    print('Dashboard created:', target)

if __name__ == '__main__':
    build_dashboard()
