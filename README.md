# Bankruptcy Prediction with Ensemble Learning

Original implementation of the core WQU DS05 workflow using data downloaded independently from UCI. Predict the `Bankrupt?` label from 95 financial variables, compare resampling strategies, tune ensemble classifiers, and communicate minority-class performance.

## Data and attribution

**Taiwanese Bankruptcy Prediction [Dataset]. (2020). UCI Machine Learning Repository. https://doi.org/10.24432/C5004D.**

Source: https://archive.ics.uci.edu/dataset/572/taiwanese+bankruptcy+prediction

License: [Creative Commons Attribution 4.0](https://creativecommons.org/licenses/by/4.0/). UCI permits sharing and adaptation with attribution. Data covers 1999–2009, contains 6,819 observations and 95 predictors, and comes from the Taiwan Economic Journal. The target follows Taiwan Stock Exchange bankruptcy definitions. Download is automatic from UCI; no WQU notebooks or private course files are distributed.

## Workflow

1. Download and cache the official ZIP; record its SHA-256 checksum.
2. Trim column names, remove exact duplicate records, separate the target, and handle nonfinite values.
3. Reserve a stratified 20% holdout with a fixed seed.
4. Compare a majority baseline, decision tree, Random Forest, and Gradient Boosting.
5. Fit median imputation and random over/undersampling **inside each training fold**. Original class proportions remain in validation and holdout data.
6. Use three-fold stratified GridSearchCV and average precision to select parameters and the final model.
7. Evaluate the holdout once at a fixed 0.5 threshold: average precision, ROC-AUC, precision, recall, F1, accuracy, and confusion matrices.
8. Export metrics, feature importances and a serialized fitted pipeline.

## Reproduce

Python 3.10 or newer:

```bash
python -m venv .venv
source .venv/bin/activate
# Windows: .venv\Scripts\activate
pip install -r requirements.txt
python train.py
python predict.py new_financial_records.csv predictions.csv
```

Training can take several minutes. Results are written to `results/`. The trained `model.joblib` is regenerated locally rather than committed. Input prediction CSVs must contain the original 95 UCI predictor columns; use their source encodings and units.

## Interpretation

Average precision matters because bankruptcy is rare; a majority classifier can have high accuracy while finding no bankrupt firms. Confusion matrices use rows=true and columns=predicted, ordered [0,1]. Model selection uses training cross-validation, never holdout scores. Tree feature importances measure model usage and are not causal effects.

This is a historical classification benchmark. The published table does not expose company identifiers and dates needed to enforce company-group or chronological splits. A random holdout therefore does not establish future forecasting performance or independence across repeated companies. Scores after resampling are not calibrated default probabilities. The 0.5 decision threshold is illustrative, not selected for a business cost function. Deployment would require temporal validation, calibration, monitoring and an agreed error-cost policy.

## Reference workflow

WorldQuant University, Project 5: Random Forests and Gradient Boosting for Bankruptcy Prediction in Poland and Taiwan: https://www.wqu.edu/data-science-lab
