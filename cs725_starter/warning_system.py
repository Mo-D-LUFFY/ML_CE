"""Independent rule-based 'warning model'. It is NOT an ML classifier."""
import json
from pathlib import Path

POLICY = json.loads((Path(__file__).parent / "policy.json").read_text(encoding="utf8"))

def assess(original: dict, candidate: dict, occupation_accessible=False):
    warnings = []
    changes = []
    for feature in original:
        if original[feature] != candidate[feature]:
            changes.append({"feature": feature, "from": native(original[feature]),
                            "to": native(candidate[feature])})
    def add(code, severity, message):
        warnings.append({"code": code, "severity": severity, "message": message})

    for feature in POLICY["immutable_in_this_experiment"]:
        if original[feature] != candidate[feature]:
            add("IMMUTABLE_FEATURE_CHANGED", "BLOCK", f"{feature} is frozen by our strict policy")

    if original["hours-per-week"] != candidate["hours-per-week"]:
        v = candidate["hours-per-week"]
        delta = v - original["hours-per-week"]
        spec = POLICY["actionable"]["hours-per-week"]
        if isinstance(v, bool) or not isinstance(v, (int, float)) or int(v) != v:
            add("INVALID_HOURS_TYPE", "BLOCK", "Weekly working hours must be an integer")
        elif not (spec["allowed_range_for_changed_value"][0] <= v <= spec["allowed_range_for_changed_value"][1]):
            add("HOURS_OUT_OF_RANGE", "BLOCK", "Changed working hours must be in [1, 65]")
        if delta < 0:
            add("WRONG_DIRECTION", "BLOCK", "Strict policy permits increasing, not decreasing, weekly hours")
        if delta > spec["maximum_increase"]:
            add("EXCESSIVE_CHANGE", "BLOCK", "Working hours may rise by at most 15 per week")
        if spec["caution_if_increase_greater_than"] < delta <= spec["maximum_increase"]:
            add("LARGE_SCHEDULE_CHANGE", "CAUTION", "A large schedule increase may not be available to this person")

    if original["occupation"] != candidate["occupation"]:
        if not occupation_accessible:
            add("OCCUPATION_REQUIRES_ACCESS", "BLOCK", "Occupation changes require explicit access / longer-horizon assumptions")
        else:
            add("OCCUPATION_UNVERIFIED", "CAUTION", "Changing occupations requires time and opportunity; feasibility is unverified")

    if any(w["severity"] == "BLOCK" for w in warnings):
        verdict = "BLOCKED"
    elif any(w["severity"] == "CAUTION" for w in warnings):
        verdict = "CAUTION"
    else:
        verdict = "PASS"
    delta_hours = max(0, candidate["hours-per-week"] - original["hours-per-week"])
    cost = delta_hours + (20 if original["occupation"] != candidate["occupation"] else 0)
    return {"verdict": verdict, "hard_constraints_satisfied": verdict != "BLOCKED",
            "warnings": warnings, "changes": changes, "change_count": len(changes),
            "estimated_cost_units": float(cost) if verdict != "BLOCKED" else None}

def native(v):
    return v.item() if hasattr(v, "item") else v
