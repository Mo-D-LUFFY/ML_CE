"""UCI Adult loader and clearly labeled synthetic Adult-like demo data."""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

COLS = ["age", "workclass", "fnlwgt", "education", "education-num", "marital-status",
        "occupation", "relationship", "race", "sex", "capital-gain", "capital-loss",
        "hours-per-week", "native-country", "income"]
FEATURES = ["age", "education-num", "hours-per-week", "capital-gain", "occupation", "marital-status"]
NUMERIC = ["age", "education-num", "hours-per-week", "capital-gain"]
CATEGORICAL = ["occupation", "marital-status"]
URLS = {
    "adult.data": "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data",
    "adult.test": "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test",
}

def load_adult(folder: Path):
    train_path, test_path = folder / "adult.data", folder / "adult.test"
    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(
            "Real UCI data missing. Download BOTH adult.data and adult.test from "
            "https://archive.ics.uci.edu/dataset/2/adult into ./data/; "
            "or run --dataset synthetic for an explicitly synthetic demo."
        )
    train = pd.read_csv(train_path, names=COLS, skipinitialspace=True, na_values=["?", " ?"])
    test = pd.read_csv(test_path, names=COLS, skipinitialspace=True, skiprows=1, na_values=["?", " ?"])
    for frame in (train, test):
        frame["income"] = frame["income"].astype(str).str.strip().str.rstrip(".")
        for col in CATEGORICAL:
            frame[col] = frame[col].str.strip()
    if not set(train.income.unique()).issubset({">50K", "<=50K"}):
        raise ValueError("Unknown UCI Adult training labels: " + repr(train.income.unique()))
    return (train[FEATURES].reset_index(drop=True), (train.income == ">50K").astype(int).reset_index(drop=True),
            test[FEATURES].reset_index(drop=True), (test.income == ">50K").astype(int).reset_index(drop=True))

def synthetic_adult(n=6000, seed=42):
    """FAKE census-like rows. Not samples of UCI Adult and not empirical evidence."""
    rng = np.random.default_rng(seed)
    age = np.clip(np.rint(rng.normal(39, 12, n)), 18, 78).astype(int)
    education = np.clip(np.rint(rng.normal(10.4, 2.6, n)), 1, 16).astype(int)
    hours = np.clip(np.rint(rng.normal(40, 10, n)), 5, 80).astype(int)
    occupation = rng.choice(["Other-service", "Sales", "Craft-repair", "Adm-clerical",
                             "Prof-specialty", "Exec-managerial", "Tech-support"],
                            n, p=[.17, .17, .15, .17, .13, .10, .11])
    marital = rng.choice(["Never-married", "Married-civ-spouse", "Divorced", "Separated"],
                         n, p=[.36, .43, .16, .05])
    gains = np.where(rng.random(n) < .08, rng.choice([2000, 5000, 10000, 20000], n), 0)
    score = (-2.3 + .039 * (age - 38) + .30 * (education - 10) + .055 * (hours - 40)
             + np.where(occupation == "Exec-managerial", .9, 0)
             + np.where(occupation == "Prof-specialty", .65, 0)
             + np.where(marital == "Married-civ-spouse", .7, 0)
             + .00020 * gains)
    p = 1 / (1 + np.exp(-score))
    label = rng.binomial(1, p)
    X = pd.DataFrame({"age": age, "education-num": education, "hours-per-week": hours,
                      "capital-gain": gains, "occupation": occupation, "marital-status": marital})
    return train_test_split(X, pd.Series(label, name="income"), test_size=.25,
                            stratify=label, random_state=seed)
