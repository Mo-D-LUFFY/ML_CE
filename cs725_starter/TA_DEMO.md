# Short TA demonstration — CS725 first milestone

**Project title:** When Is a Counterfactual Actually Actionable?

**Research question:** When we restrict counterfactual suggestions to
changes permitted by explicit actionability rules, how does the number of
model-flipping, feasible suggestions change?

## One-minute live demonstration

1. Open **`demo_dashboard.html`** in a browser: select a classifier and an individual case.
   The dashboard is generated from the trained-model outputs and works offline.
   To reproduce it, run `python run_demo.py --dataset synthetic` (or `--dataset adult` with the downloaded official files).
2. Open `outputs/classifier_metrics.json`. These are actual holdout scores
   for the **named dataset mode**.
3. Open `outputs/demo_example.json` and point out:
   - the starting prediction and its model probability;
   - a prediction-flipping suggestion rejected with a specific reason;
   - a possible schedule change and its independent caution/constraint result.
4. Open `outputs/comparison_summary.csv` to see counts on 100 selected held-out
   negative examples (or the number actually available).
5. Show `policy.json`: rules are machine readable, independently tested.

**Exactly what is a warning?** It is a **rule-based actionability flag**,
not a second ML model. `BLOCKED`: policy is violated. `CAUTION`: no hard rule
is broken, but real-world feasibility is not verified. `PASS`: no implemented
rule flagged the profile. The prediction flip is reported independently.

**What is novel here?** Not a novel predictive classifier: the focus is a
reproducible evaluation of the gap between changing a model's answer and
suggesting an admissible real-world action. The final study will add DiCE,
matched experimental search budgets, plausible-data checks and sensitivity
analysis.

**Critical truth about today's files:** The included demonstration metrics
are based on **synthetic Adult-like records**, not the historical UCI Adult
records. They prove that the code runs, not that the hypotheses hold on real
observations. See `README.md` to switch to actual Adult data.
