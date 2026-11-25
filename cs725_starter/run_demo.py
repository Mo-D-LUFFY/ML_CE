"""Train two ML models, create counterfactuals, validate warnings, save JSON/CSV.

Usage: python run_demo.py --dataset synthetic
       python run_demo.py --dataset adult    # after placing official adult.data/test in data/
"""
import argparse
import json
import time
from pathlib import Path
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, confusion_matrix,
                             f1_score, precision_score, recall_score, roc_auc_score)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from data_pipeline import (NUMERIC, CATEGORICAL, load_adult, synthetic_adult)
from counterfactual import as_record, generate
from warning_system import POLICY, assess

ROOT = Path(__file__).parent
OUT = ROOT / "outputs"
MODELS = ROOT / "models"

def dump(path, content):
    path.write_text(json.dumps(content, indent=2, allow_nan=False), encoding="utf8")

def make_model(model_type):
    transformer = ColumnTransformer([
        ("numeric", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), NUMERIC),
        ("categorical", Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                                  ("onehot", OneHotEncoder(handle_unknown="ignore"))]), CATEGORICAL)
    ])
    classifier = (LogisticRegression(max_iter=1200, random_state=42)
                  if model_type == "logistic_regression" else
                  RandomForestClassifier(n_estimators=120, min_samples_leaf=5,
                                         random_state=42, n_jobs=-1))
    return Pipeline([("preprocess", transformer), ("classifier", classifier)])

def metrics(model, X_test, y_test):
    yp = model.predict(X_test)
    prob = model.predict_proba(X_test)[:, 1]
    tn, fp, fn, tp = confusion_matrix(y_test, yp, labels=[0, 1]).ravel()
    return {"n_test": int(len(y_test)), "positive_test_prevalence": round(float(y_test.mean()), 4),
            "accuracy": round(float(accuracy_score(y_test, yp)), 4),
            "balanced_accuracy": round(float(balanced_accuracy_score(y_test, yp)), 4),
            "positive_precision": round(float(precision_score(y_test, yp, zero_division=0)), 4),
            "positive_recall": round(float(recall_score(y_test, yp, zero_division=0)), 4),
            "positive_f1": round(float(f1_score(y_test, yp, zero_division=0)), 4),
            "roc_auc": round(float(roc_auc_score(y_test, prob)), 4),
            "confusion_matrix": {"TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp)}}

def make_experiment(model, X_test, model_name, n_queries=100):
    predictions = model.predict(X_test)
    negatives = X_test.loc[predictions == 0]
    sampled = negatives.sample(min(n_queries, len(negatives)), random_state=42)
    rows, examples = [], []
    for idx, original in sampled.iterrows():
        orig = as_record(original.to_dict())
        result = {method: generate(model, orig, method) for method in ("unconstrained", "constrained")}
        examples.append({"query_id": str(idx), "original": orig, **result})
        for method, record in result.items():
            rows.append({"classifier": model_name, "query_id": str(idx), "generator": method,
                         "flip_success": int(record is not None),
                         "feasible_flip": int(record is not None and record["hard_constraints_satisfied"]),
                         "blocked": int(record is not None and record["verdict"] == "BLOCKED"),
                         "caution": int(record is not None and record["verdict"] == "CAUTION"),
                         "cost_units": record["estimated_cost_units"] if record else None,
                         "edits": record["change_count"] if record else None,
                         "proximity": record["normalized_change_distance"] if record else None})
    frame = pd.DataFrame(rows)
    summary = []
    for method, group in frame.groupby("generator", sort=False):
        total = len(group)
        count_flip = int(group.flip_success.sum())
        count_feasible = int(group.feasible_flip.sum())
        summary.append({"classifier": model_name, "generator": method,
                        "queries": total, "flip_success_count": count_flip,
                        "feasible_flip_count": count_feasible,
                        "flip_success_rate": round(count_flip / total, 3),
                        "feasible_valid_coverage": round(count_feasible / total, 3),
                        "feasibility_among_flips": round(count_feasible / count_flip, 3) if count_flip else None,
                        "blocked_candidate_count": int(group.blocked.sum()),
                        "median_cost_for_feasible": (
                            round(float(group.loc[group.feasible_flip == 1, "cost_units"].median()), 3)
                            if count_feasible else None)})
    return frame, summary, examples

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["synthetic", "adult"], default="synthetic")
    parser.add_argument("--queries", type=int, default=100)
    args = parser.parse_args()
    OUT.mkdir(exist_ok=True)
    MODELS.mkdir(exist_ok=True)
    if args.dataset == "adult":
        X_train, y_train, X_test, y_test = load_adult(ROOT / "data")
    else:
        X_train, X_test, y_train, y_test = synthetic_adult(seed=42)
        sample = X_train.copy()
        sample["target_gt_50k"] = y_train.to_numpy()
        sample.head(150).to_csv(ROOT / "data" / "synthetic_adult_preview.csv", index=False)
    metadata = {"dataset": args.dataset, "is_synthetic": args.dataset == "synthetic",
                "n_train": len(X_train), "n_test": len(X_test), "policy_id": POLICY["policy_id"],
                "target": "1=>50K, 0=<=50K", "random_seed": 42,
                "interpretation_warning": POLICY["warning"],
                "generation_note": "Small auditable grid search, NOT DiCE or globally optimal counterfactuals"}
    dump(OUT / "run_metadata.json", metadata)
    records, summaries, case_examples = [], [], {}
    all_metrics = {}
    for name in ("logistic_regression", "random_forest"):
        model = make_model(name)
        begin = time.perf_counter()
        model.fit(X_train, y_train)
        train_seconds = round(time.perf_counter() - begin, 3)
        scores = metrics(model, X_test, y_test)
        scores["train_seconds"] = train_seconds
        all_metrics[name] = scores
        joblib.dump(model, MODELS / f"{args.dataset}_{name}.joblib")
        table, summary, cases = make_experiment(model, X_test, name, args.queries)
        records.append(table)
        summaries.extend(summary)
        case_examples[name] = cases
        print(f"{name}: test accuracy={scores['accuracy']:.3f}, F1={scores['positive_f1']:.3f}, AUC={scores['roc_auc']:.3f}")
        for s in summary:
            print(f"  {s['generator']}: {s['flip_success_count']}/{s['queries']} flips, "
                  f"{s['feasible_flip_count']}/{s['queries']} feasible flips")
    dump(OUT / "classifier_metrics.json", all_metrics)
    dump(OUT / "comparison_summary.json", summaries)
    dump(OUT / "query_examples.json", case_examples)
    pd.concat(records, ignore_index=True).to_csv(OUT / "per_query_results.csv", index=False)
    pd.DataFrame(summaries).to_csv(OUT / "comparison_summary.csv", index=False)
    # Prefer a single held-out row where constrained works, with an invalid competing flip.
    choices = case_examples["logistic_regression"]
    selected = next((e for e in choices if e["constrained"] and e["unconstrained"]
                     and e["unconstrained"]["verdict"] == "BLOCKED"), None)
    if selected is None:
        selected = next((e for e in choices if e["constrained"]), None)
    if selected is None:
        selected = choices[0] if choices else None
    if selected:
        chosen = {"query_id": selected["query_id"], "dataset": args.dataset,
                  "is_synthetic": args.dataset == "synthetic",
                  "classifier": "logistic_regression", "original": selected["original"],
                  "unconstrained_counterfactual": selected["unconstrained"],
                  "constrained_counterfactual": selected["constrained"]}
        # Explicitly inspect a specified invalid edit, even if it doesn't flip.
        manual = {**selected["original"], "age": selected["original"]["age"] + 20}
        manual_p = float(joblib.load(MODELS / f"{args.dataset}_logistic_regression.joblib")
                         .predict_proba(pd.DataFrame([manual]))[0, 1])
        chosen["manual_age_change_warning"] = {
            "proposed": manual, "probability_gt_50k": round(manual_p, 5),
            "flips_prediction": manual_p >= 0.5, **assess(selected["original"], manual)}
        dump(OUT / "demo_example.json", chosen)
    from make_dashboard import build_dashboard
    build_dashboard()
    print(f"Saved outputs in {OUT}")

if __name__ == "__main__":
    main()
