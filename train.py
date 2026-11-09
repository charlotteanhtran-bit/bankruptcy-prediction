"""Original DS05-inspired bankruptcy benchmark; run with python train.py."""
from pathlib import Path
import io, json, zipfile, hashlib
from urllib.request import urlopen
import numpy as np
import pandas as pd
import joblib
from sklearn.dummy import DummyClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.metrics import (average_precision_score, roc_auc_score, precision_score,
                            recall_score, f1_score, accuracy_score, confusion_matrix)
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

ROOT = Path(__file__).resolve().parent
URL = 'https://archive.ics.uci.edu/static/public/572/taiwanese+bankruptcy+prediction.zip'

def main():
    data, out = ROOT/'data', ROOT/'results'
    data.mkdir(exist_ok=True); out.mkdir(exist_ok=True)
    archive = data/'source.zip'
    if not archive.exists():
        with urlopen(URL, timeout=120) as response:
            archive.write_bytes(response.read())
    with zipfile.ZipFile(archive) as z:
        name = next(n for n in z.namelist() if n.endswith('data.csv'))
        df = pd.read_csv(io.BytesIO(z.read(name)))
    df.columns = df.columns.str.strip()
    original_rows = len(df)
    df = df.drop_duplicates()
    y = df.pop('Bankrupt?').astype(int)
    X = df.apply(pd.to_numeric, errors='coerce').replace([np.inf, -np.inf], np.nan)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=.2, stratify=y, random_state=42)
    cv = StratifiedKFold(3, shuffle=True, random_state=42)
    specs = {
        'majority_baseline': (DummyClassifier(strategy='most_frequent'), {}),
        'decision_tree': (DecisionTreeClassifier(random_state=42), {'model__max_depth':[3, 6]}),
        'random_forest': (RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=1),
                          {'model__max_depth':[6, None], 'model__min_samples_leaf':[2, 5]}),
        'gradient_boosting': (GradientBoostingClassifier(n_estimators=100, random_state=42),
                              {'model__max_depth':[1, 2], 'model__learning_rate':[.05, .1]})}
    rows, fitted = [], {}
    for name, (model, grid) in specs.items():
        pipe = Pipeline([('imputer',SimpleImputer(strategy='median')),
                         ('sampler','passthrough'), ('model',model)])
        if name != 'majority_baseline':
            grid['sampler'] = ['passthrough', RandomUnderSampler(random_state=42), RandomOverSampler(random_state=42)]
        search = GridSearchCV(pipe, grid, scoring='average_precision', cv=cv, n_jobs=2, refit=True)
        search.fit(Xtr, ytr)
        fitted[name] = search.best_estimator_
        rows.append({'model':name, 'cv_average_precision':float(search.best_score_),
                     'parameters':{k:str(v) for k,v in search.best_params_.items()}})
        print(name, search.best_score_, flush=True)
    winner = max(rows, key=lambda r:r['cv_average_precision'])['model']
    # Winner chosen using training CV only; held-out labels never guide selection.
    for row in rows:
        model = fitted[row['model']]
        prob = model.predict_proba(Xte)[:,1]; pred = (prob >= .5).astype(int)
        row.update(average_precision=float(average_precision_score(yte,prob)),
                   roc_auc=float(roc_auc_score(yte,prob)), accuracy=float(accuracy_score(yte,pred)),
                   precision=float(precision_score(yte,pred,zero_division=0)),
                   recall=float(recall_score(yte,pred,zero_division=0)),
                   f1=float(f1_score(yte,pred,zero_division=0)), confusion_matrix=confusion_matrix(yte,pred).tolist())
    joblib.dump(fitted[winner], out/'model.joblib')
    importances = fitted[winner].named_steps['model'].feature_importances_
    pd.Series(importances,index=X.columns).sort_values(ascending=False).rename('importance').to_csv(out/'feature_importance.csv')
    result = {'source_url':URL, 'sha256':hashlib.sha256(archive.read_bytes()).hexdigest(),
              'original_rows':original_rows, 'unique_rows':len(X), 'features':X.shape[1],
              'train_rows':len(Xtr), 'test_rows':len(Xte), 'train_bankrupt':int(ytr.sum()),
              'test_bankrupt':int(yte.sum()), 'test_prevalence':float(yte.mean()),
              'selected_model':winner, 'threshold':.5, 'models':rows}
    (out/'metrics.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))

if __name__ == '__main__':
    main()
