"""Show a prediction, candidate counterfactuals, and warnings for a JSON profile.

python inspect_profile.py --profile example_profile.json --dataset synthetic
"""
import argparse
import json
from pathlib import Path
import joblib
import pandas as pd
from counterfactual import generate
from warning_system import POLICY

ROOT = Path(__file__).parent

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--profile', default='example_profile.json')
    p.add_argument('--dataset', choices=['synthetic', 'adult'], default='synthetic')
    p.add_argument('--model', choices=['logistic_regression', 'random_forest'], default='logistic_regression')
    args = p.parse_args()
    x = json.loads(Path(args.profile).read_text(encoding='utf8'))
    if set(x) != {"age", "education-num", "hours-per-week", "capital-gain", "occupation", "marital-status"}:
        p.error('Profile must have exactly the six feature keys listed in example_profile.json')
    file = ROOT / 'models' / f'{args.dataset}_{args.model}.joblib'
    if not file.exists():
        p.error(f'{file} not found. Run python run_demo.py --dataset {args.dataset} first.')
    model = joblib.load(file)
    probability = float(model.predict_proba(pd.DataFrame([x]))[0, 1])
    output = {"dataset": args.dataset, "classifier": args.model,
              "input_profile": x, "original_prediction": '>50K' if probability >= .5 else '<=50K',
              "probability_gt_50k": round(probability, 5),
              "unconstrained": generate(model, x, 'unconstrained') if probability < .5 else None,
              "constrained": generate(model, x, 'constrained') if probability < .5 else None,
              "policy_id": POLICY['policy_id'],
              "warning": POLICY['warning']}
    print(json.dumps(output, indent=2))
    out = ROOT / 'outputs' / 'custom_prediction.json'
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(output, indent=2), encoding='utf8')
    print(f'\nSaved {out}')

if __name__ == '__main__':
    main()
