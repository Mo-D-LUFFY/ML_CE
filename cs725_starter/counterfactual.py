"""Simple auditable grid search, a starter alternative to DiCE."""
import itertools
import pandas as pd
from warning_system import assess, native

TARGET_THRESHOLD = 0.5

def as_record(row):
    return {key: native(value) for key, value in dict(row).items()}

def propose(original, method="unconstrained"):
    orig = as_record(original)
    proposals = []
    if method == "constrained":
        for increase in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]:
            new_hours = orig["hours-per-week"] + increase
            if new_hours <= 65:
                proposals.append({**orig, "hours-per-week": int(new_hours)})
    elif method == "unconstrained":
        for key, targets in [
            ("age", [orig["age"] + x for x in [5, 10, 20, 30]]),
            ("education-num", [orig["education-num"] + x for x in [2, 4, 6]] + [16]),
            ("hours-per-week", [orig["hours-per-week"] + x for x in [5, 10, 15, 25, 35]]),
            ("capital-gain", [5000, 10000, 20000, 35000]),
            ("occupation", ["Exec-managerial", "Prof-specialty", "Sales"]),
            ("marital-status", ["Married-civ-spouse"]),
        ]:
            for target in targets:
                c = {**orig, key: target}
                if key in ("age", "education-num", "hours-per-week", "capital-gain"):
                    bounds = {"age": (18, 90), "education-num": (1, 16),
                              "hours-per-week": (1, 99), "capital-gain": (0, 99999)}
                    lo, hi = bounds[key]
                    c[key] = int(max(lo, min(hi, target)))
                if c != orig:
                    proposals.append(c)
        # Limited two-feature search improves coverage while keeping runs cheap.
        for a, b in itertools.product([10, 20, 30], [5, 15, 25]):
            proposals.append({**orig, "age": min(90, orig["age"] + a),
                              "hours-per-week": min(99, orig["hours-per-week"] + b)})
        for e, h in itertools.product([2, 4, 6], [10, 20]):
            proposals.append({**orig, "education-num": min(16, orig["education-num"] + e),
                              "hours-per-week": min(99, orig["hours-per-week"] + h)})
    else:
        raise ValueError("Unknown generator: " + method)
    return list({tuple(sorted(c.items())): c for c in proposals}.values())

def measure_distance(a, b):
    scale = {"age": 20, "education-num": 5, "hours-per-week": 15, "capital-gain": 10000}
    return sum((abs(b[k] - a[k]) / scale[k] if k in scale else 1)
               for k in a if a[k] != b[k])

def generate(model, original, method="unconstrained"):
    orig = as_record(original)
    before = float(model.predict_proba(pd.DataFrame([orig]))[0, 1])
    proposals = propose(orig, method)
    if not proposals:
        return None
    probs = model.predict_proba(pd.DataFrame(proposals))[:, 1]
    valid = []
    for candidate, p in zip(proposals, probs):
        if p >= TARGET_THRESHOLD and before < TARGET_THRESHOLD:
            warning = assess(orig, candidate)
            if method == "constrained" and not warning["hard_constraints_satisfied"]:
                continue
            valid.append((measure_distance(orig, candidate), warning["change_count"],
                          candidate, float(p), warning))
    if not valid:
        return None
    valid.sort(key=lambda x: (x[0], x[1], x[3]))
    dist, changes, candidate, after, warning = valid[0]
    return {"original": orig, "original_prediction": "<=50K", "original_probability_gt_50k": round(before, 5),
            "counterfactual": candidate, "counterfactual_prediction": ">50K",
            "counterfactual_probability_gt_50k": round(after, 5), "prediction_flipped": True,
            "generator": method, "normalized_change_distance": round(dist, 4),
            **warning}
