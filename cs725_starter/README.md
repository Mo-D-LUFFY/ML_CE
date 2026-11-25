# CS725 — Actionable Counterfactual Starter Prototype (v0.1)

## Goal

For a **fixed binary income-classification model**, generate changes that switch its
prediction from `<=50K` to `>50K`, then separately check whether proposed edits
obey a declared actionability policy. This tests **model-based counterfactual
feasibility**; it does **not** prove that following any suggestion will change real income.

## Three things to show your TA today

1. Two trained predictors: **Logistic Regression** and **Random Forest**.
   Open `demo_dashboard.html` in your browser for a standalone interactive view of actual generated outputs.
2. For the **same held-out negative profile**, an **unconstrained** candidate
   can flip the model but **fail** the actionability rules, while a **constrained**
   candidate may obey them (with caution, if necessary).
3. Machine-readable per-query JSON and a CSV comparison of methods.

## Quick start

In this directory, on Windows PowerShell or WSL/Linux:

```bash
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python run_demo.py --dataset synthetic
python -m pytest -q
python inspect_profile.py --profile example_profile.json --dataset synthetic
deactivate
```

The `synthetic` setting trains on **6000 fabricated Adult-like examples** created
by `data_pipeline.py` using an illustrative data-generating formula. These are
**not** copied from Adult or its original statistical distribution. Results on
this setting are a software demonstration only, **not real experimental results**.

### Switch to the real Adult dataset

1. Visit https://archive.ics.uci.edu/dataset/2/adult
2. Download original files **adult.data** and **adult.test** into the `data/` folder.
   (If the UCI download is a zip, extract these two files.)
3. Run:

```bash
python run_demo.py --dataset adult
python inspect_profile.py --profile example_profile.json --dataset adult
```

The code uses the **official train/test files**, with missing-category imputation
and one-hot encoding fitted **only on training rows**. The real data never
silently fall back to fabricated data.

Source credit: Becker, B. & Kohavi, R. (1996). *Adult*, UCI Machine Learning
Repository. https://doi.org/10.24432/C5XW20 (CC BY 4.0).

## The input feature schema

We deliberately use **6 of Adult's 14 inputs** to keep v0.1 understandable:

| Feature | Type | Treatment under strict policy |
|---|---|---|
| age | integer | immutable *at the decision time* |
| education-num | integer | current recorded education; immutable in short horizon |
| hours-per-week | integer | prospective schedule increases up to 15, resulting value <=65 |
| capital-gain | integer | recorded value, immutable for current decision |
| occupation | categorical | conditional: requires explicit career-access assumption |
| marital-status | categorical | immutable in this experiment |

Excluded: `fnlwgt` (survey weight), sensitive demographics `race`/`sex`,
redundant `education` because `education-num` is already used, plus other
columns to keep the starter small. Exclusion is **not** proof of fairness.

## Model definitions

- **Logistic Regression** (`scikit-learn`): standardize numerical variables,
  one-hot-encode categories, fit binary classifier.
- **Random Forest** (`scikit-learn`): 120 trees, min leaf size 5, same inputs.
- **Warning system**: transparent **rules**, not a trained predictor of real
  consequences. `BLOCKED` means a hard policy violation; `CAUTION` means hard
  rules pass but there's an unverified practical limitation; `PASS` means no
  rule in this simplified policy flagged the suggestion.

All outputs use class 1 = `>50K`, class 0 = `<=50K`; threshold = 0.5.
**The probability refers to the trained classifier's model score**, not a
validated individual's probability of achieving higher income.

## Outputs

| File | Contents |
|---|---|
| `demo_dashboard.html` | Offline interactive report with 100 query examples per classifier |
| `outputs/run_metadata.json` | Dataset identity, synthetic flag, train/test counts |
| `outputs/classifier_metrics.json` | Held-out accuracy, F1, ROC AUC, confusion matrix |
| `outputs/demo_example.json` | One clearly identified held-out example with both methods and warnings |
| `outputs/comparison_summary.csv` | Aggregate flip and feasible flip counts / rates |
| `outputs/per_query_results.csv` | Same held-out negative queries for both generation methods |
| `outputs/custom_prediction.json` | A profile supplied to `inspect_profile.py` |
| `models/*joblib` | Saved trained scikit-learn pipelines |
| `data/synthetic_adult_preview.csv` | Synthetic examples for spreadsheet inspection |

## Sample output fields

```json
{
  "original_prediction": "<=50K",
  "original_probability_gt_50k": 0.35,
  "counterfactual_prediction": ">50K",
  "counterfactual_probability_gt_50k": 0.51,
  "prediction_flipped": true,
  "changes": [{"feature": "hours-per-week", "from": 35, "to": 46}],
  "verdict": "CAUTION",
  "hard_constraints_satisfied": true,
  "warnings": [{"code": "LARGE_SCHEDULE_CHANGE", "severity": "CAUTION",
                "message": "A large schedule increase may not be available to this person"}],
  "estimated_cost_units": 11.0
}
```

The numbers above are illustrative. Use the **generated** JSON for measured output.
The units in `estimated_cost_units` are arbitrary comparison penalties, not money.
A `BLOCKED` candidate has `estimated_cost_units: null` because its cost is not a
meaningful actionable quantity under this policy.

When no model-flipping candidate exists within the defined search, the output
is `null`: do not invent a result. `null` is a **search failure**, not a proof
that no feasible counterfactual exists globally.

## Research limitation and next step

This is a **baseline prototype**, not a finished research comparison. The two
searches explore **different candidate spaces and effort budgets**; compare
rates as an initial demonstration only, **not** evidence that explicit
constraints alone caused the observed differences.

Next: use **the same DiCE generator**, the same held-out negative inputs,
matched candidate/time budgets, independent validator, predeclared strict
and relaxed policies, then add a plausibility proxy and sensitivity analysis.
A real-world causal effect is not identified by this experiment.
