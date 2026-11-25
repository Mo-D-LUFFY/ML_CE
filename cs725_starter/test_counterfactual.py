import pandas as pd
from counterfactual import propose, generate
from warning_system import assess

BASE = {"age": 40, "education-num": 13, "hours-per-week": 35,
        "capital-gain": 0, "occupation": "Sales", "marital-status": "Never-married"}

class DummyModel:
    def predict_proba(self, frame):
        # A deliberately tiny deterministic model for generator unit testing.
        score = frame['hours-per-week'].to_numpy(dtype=float)
        p = (score >= 45).astype(float) * .9 + (score < 45).astype(float) * .1
        import numpy as np
        return np.column_stack([1-p, p])

def test_constrained_generator_keeps_immutable_features():
    record = generate(DummyModel(), BASE, 'constrained')
    assert record is not None and record['prediction_flipped']
    assert record['counterfactual']['hours-per-week'] == 45
    assert record['hard_constraints_satisfied']
    assert record['change_count'] == 1

def test_generator_asks_only_allowed_hours():
    for rec in propose(BASE, 'constrained'):
        assert assess(BASE, rec)['hard_constraints_satisfied']

def test_no_constrained_solution_reports_none():
    base = {**BASE, 'hours-per-week': 65}
    assert generate(DummyModel(), base, 'constrained') is None
