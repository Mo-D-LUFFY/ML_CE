"""Milestone 1: official Adult split, 6/14-feature ablation, LR/RF/XGBoost."""
import argparse
import hashlib
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
import xgboost
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, precision_score,
    recall_score, f1_score, roc_auc_score, average_precision_score, brier_score_loss,
    confusion_matrix)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parent
ALL = ['age','workclass','fnlwgt','education','education-num','marital-status',
       'occupation','relationship','race','sex','capital-gain','capital-loss',
       'hours-per-week','native-country']
SIX = ['age','education-num','hours-per-week','capital-gain','occupation','marital-status']
NUM = ['age','fnlwgt','education-num','capital-gain','capital-loss','hours-per-week']


def load(data_dir, smoke=False):
    paths = [data_dir / 'adult.data', data_dir / 'adult.test']
    frames = []
    for path, skip in zip(paths, [0, 1]):
        df = pd.read_csv(path, names=ALL + ['income'], skiprows=skip,
                         skipinitialspace=True, na_values=['?'])
        for col in set(ALL) - set(NUM):
            df[col] = df[col].str.strip()
        df['income'] = df['income'].str.strip().str.rstrip('.')
        if not df['income'].isin(['<=50K','>50K']).all():
            raise ValueError(f'Invalid target labels in {path}')
        frames.append(df)
    counts = [len(df) for df in frames]
    if not smoke and counts != [32561, 16281]:
        raise ValueError(f'Expected official Adult split, found {counts}')
    return frames, [hashlib.sha256(p.read_bytes()).hexdigest() for p in paths]


def make_model(name, features, seed, jobs):
    numeric = [c for c in features if c in NUM]
    categorical = [c for c in features if c not in NUM]
    pre = ColumnTransformer([
        ('num', make_pipeline(SimpleImputer(strategy='median'), StandardScaler()), numeric),
        ('cat', make_pipeline(SimpleImputer(strategy='constant', fill_value='MISSING'),
                             OneHotEncoder(handle_unknown='ignore')), categorical)])
    if name == 'lr':
        clf = LogisticRegression(max_iter=1500, random_state=seed)
    elif name == 'rf':
        clf = RandomForestClassifier(n_estimators=120, min_samples_leaf=5,
                                     random_state=seed, n_jobs=jobs)
    else:
        clf = XGBClassifier(n_estimators=300, max_depth=5, learning_rate=.08,
                            subsample=.85, colsample_bytree=.85, reg_lambda=1,
                            tree_method='hist', eval_metric='logloss',
                            random_state=seed, n_jobs=jobs)
    return make_pipeline(pre, clf)


def scores(y, p):
    pred = (p >= .5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0,1]).ravel()
    return dict(accuracy=accuracy_score(y,pred), balanced_accuracy=balanced_accuracy_score(y,pred),
        precision=precision_score(y,pred,zero_division=0), recall=recall_score(y,pred,zero_division=0),
        f1=f1_score(y,pred,zero_division=0), roc_auc=roc_auc_score(y,p),
        average_precision=average_precision_score(y,p), brier=brier_score_loss(y,p),
        tn=int(tn),fp=int(fp),fn=int(fn),tp=int(tp))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data-dir', type=Path, default=ROOT.parent/'cs725_starter'/'data')
    ap.add_argument('--out', type=Path, default=ROOT/'results'/'classical')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--jobs', type=int, default=4)
    ap.add_argument('--smoke', action='store_true', help='Allow fixture row counts; not report data')
    args = ap.parse_args()
    (train,test), hashes = load(args.data_dir,args.smoke)
    out=args.out; out.mkdir(parents=True,exist_ok=True)
    (out/'models').mkdir(exist_ok=True)
    y=(train.income=='>50K').astype(int); yt=(test.income=='>50K').astype(int)
    fit_idx,val_idx=train_test_split(np.arange(len(train)),test_size=.2,stratify=y,random_state=args.seed)
    meta=dict(dataset='fixture' if args.smoke else 'UCI Adult official split',
              n_train=len(train),n_validation=len(val_idx),n_test=len(test),seed=args.seed,
              threshold=.5,feature_sets={'six':SIX,'full':ALL},sha256=hashes,
              sklearn=sklearn.__version__,xgboost=xgboost.__version__,
              protocol='Fixed initial hyperparameters; validation fit then refit on all official training rows; no test tuning')
    (out/'metadata.json').write_text(json.dumps(meta,indent=2),encoding='utf8')
    rows=[]; predictions=pd.DataFrame({'row_id':np.arange(len(test)), 'target':yt})
    for tag,features in [('six',SIX),('full',ALL)]:
        for name in ['lr','rf','xgb']:
            key=f'{tag}_{name}'; print(f'Training {key} ...',flush=True)
            model=make_model(name,features,args.seed,args.jobs)
            model.fit(train.iloc[fit_idx][features],y.iloc[fit_idx])
            pv=model.predict_proba(train.iloc[val_idx][features])[:,1]
            rows.append(dict(model=name,features=tag,split='validation',n=len(val_idx),**scores(y.iloc[val_idx],pv)))
            model=make_model(name,features,args.seed,args.jobs)
            t=time.perf_counter(); model.fit(train[features],y); seconds=time.perf_counter()-t
            pt=model.predict_proba(test[features])[:,1]
            rows.append(dict(model=name,features=tag,split='test',n=len(test),train_seconds=seconds,**scores(yt,pt)))
            predictions[key]=pt
            joblib.dump(model,out/'models'/f'{key}.joblib',compress=3)
            pd.DataFrame(rows).to_csv(out/'metrics.csv',index=False)
            predictions.to_csv(out/'test_predictions.csv',index=False)
            print(f'  test F1={rows[-1]["f1"]:.4f} AUC={rows[-1]["roc_auc"]:.4f} fit={seconds:.1f}s',flush=True)
    print(f'Results: {out.resolve()}')


if __name__ == '__main__':
    main()
