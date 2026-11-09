"""Apply the saved pipeline to a CSV in the original UCI feature schema."""
import sys
from pathlib import Path
import joblib
import pandas as pd
import numpy as np

if __name__ == '__main__':
    if len(sys.argv) != 3:
        raise SystemExit('Usage: python predict.py input.csv output.csv')
    model = joblib.load(Path(__file__).resolve().parent/'results/model.joblib')
    data = pd.read_csv(sys.argv[1]); data.columns = data.columns.str.strip()
    columns = list(model.feature_names_in_)
    missing = sorted(set(columns)-set(data.columns))
    if missing:
        raise ValueError(f'Missing predictor columns: {missing}')
    X = data[columns].apply(pd.to_numeric,errors='raise').replace([np.inf,-np.inf],np.nan)
    scores = model.predict_proba(X)[:,1]
    pd.DataFrame({'bankruptcy_score':scores,'predicted_bankrupt':(scores>=.5).astype(int)}).to_csv(sys.argv[2],index=False)
