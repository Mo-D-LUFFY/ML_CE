from warning_system import assess

BASE = {"age": 36, "education-num": 10, "hours-per-week": 38,
        "capital-gain": 0, "occupation": "Sales", "marital-status": "Never-married"}

def test_age_edit_blocked():
    x = {**BASE, "age": 46}
    r = assess(BASE, x)
    assert r["verdict"] == "BLOCKED"
    assert "IMMUTABLE_FEATURE_CHANGED" in [w["code"] for w in r["warnings"]]

def test_allow_small_hours_change():
    assert assess(BASE, {**BASE, "hours-per-week": 43})["verdict"] == "PASS"

def test_hours_too_large_blocked():
    assert assess(BASE, {**BASE, "hours-per-week": 75})["verdict"] == "BLOCKED"

def test_hours_large_caution():
    assert assess(BASE, {**BASE, "hours-per-week": 50})["verdict"] == "CAUTION"

def test_occupation_requires_explicit_access():
    proposal = {**BASE, "occupation": "Prof-specialty"}
    assert assess(BASE, proposal)["verdict"] == "BLOCKED"
    assert assess(BASE, proposal, occupation_accessible=True)["verdict"] == "CAUTION"

def test_unmodified_profile_passes():
    assert assess(BASE, BASE)["verdict"] == "PASS"
