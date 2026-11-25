"""Exact tiny counterfactual example for the CS725 beginner guide.

All coefficients and effort scores are deliberately made up for teaching.
This script does NOT train an ML model or provide real student advice.

Run:
    python toy_counterfactual.py
"""

from __future__ import annotations
from dataclasses import asdict, dataclass
from itertools import product
import json
from pathlib import Path


@dataclass(frozen=True)
class Profile:
    study_hours: int
    extra_support: int
    earlier_grade: int


def model_score(p: Profile) -> int:
    return 2 * p.study_hours + 3 * p.extra_support + 2 * p.earlier_grade


def model_label(p: Profile) -> int:
    return int(model_score(p) >= 14)


def constraint_reasons(original: Profile, candidate: Profile, support_accessible: bool = True) -> list[str]:
    reasons = []
    if candidate.earlier_grade != original.earlier_grade:
        reasons.append("earlier_grade cannot be changed retroactively")
    if not (0 <= candidate.study_hours <= 4):
        reasons.append("study_hours must be in [0, 4]")
    if candidate.study_hours < original.study_hours:
        reasons.append("study_hours may only increase")
    if candidate.study_hours - original.study_hours > 2:
        reasons.append("study_hours may increase by at most 2")
    if candidate.extra_support not in (0, 1):
        reasons.append("extra_support must equal 0 or 1")
    if original.extra_support == 1 and candidate.extra_support == 0:
        reasons.append("support may not be removed in this toy policy")
    if not support_accessible and candidate.extra_support > original.extra_support:
        reasons.append("extra_support is not available")
    return reasons


def cost(original: Profile, candidate: Profile) -> int:
    return (candidate.study_hours - original.study_hours) + 3 * (
        candidate.extra_support - original.extra_support
    )


def generate_feasible_counterfactuals(original: Profile, support_accessible: bool = True) -> list[dict]:
    found = []
    for study_hours, support in product(range(5), (0, 1)):
        candidate = Profile(study_hours, support, original.earlier_grade)
        reasons = constraint_reasons(original, candidate, support_accessible)
        if not reasons and model_label(candidate) == 1:
            found.append({
                "candidate": asdict(candidate),
                "score": model_score(candidate),
                "predicted_label": model_label(candidate),
                "illustrative_cost_units": cost(original, candidate),
                "feasible": True,
            })
    return sorted(found, key=lambda item: (item["illustrative_cost_units"], item["candidate"]["study_hours"]))


def run() -> None:
    original = Profile(2, 0, 3)
    print(f"Original: {original} -> score {model_score(original)} -> class {model_label(original)}")
    print("\nManually specified candidates:")
    examples = (
        Profile(2, 0, 5),  # impossible old-grade edit
        Profile(4, 0, 3),  # feasible study change
        Profile(3, 1, 3),  # feasible with available support
        Profile(9, 0, 3),  # forbidden study-hours value
    )
    for p in examples:
        problems = constraint_reasons(original, p)
        print(f"  {p} -> score {model_score(p):2} -> flip {bool(model_label(p))!s:5} -> "
              f"{'REJECT: ' + '; '.join(problems) if problems else 'ALLOWED'}")
    feasible = generate_feasible_counterfactuals(original)
    print("\nFeasible prediction-flipping candidates, in increasing cost order:")
    for c in feasible:
        print(f"  {c['candidate']} | score={c['score']} | cost={c['illustrative_cost_units']}")

    result = {
        "warning": "Synthetic teaching example, not real student advice",
        "original": asdict(original),
        "original_score": model_score(original),
        "original_prediction": model_label(original),
        "accepted_counterfactuals": feasible,
    }
    path = Path(__file__).parent / "toy_candidates.json"
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"\nSaved reproducible result to {path}")


if __name__ == "__main__":
    run()
