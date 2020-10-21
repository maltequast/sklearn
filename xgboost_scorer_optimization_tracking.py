import optuna
from optuna.integration.mlflow import MLflowCallback
from optuna.integration import SkoptSampler
import xgboost as xgb
from sklearn.model_selection import cross_validate
from sklearn.metrics import make_scorer, precision_score, recall_score, f1_score

import numpy as np
import mlflow
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

scoring = {'accuracy': 'accuracy',
           'precision': make_scorer(precision_score, average='weighted'),
           'recall': make_scorer(recall_score, average='weighted'),
           'f1': make_scorer(f1_score, average='weighted'),
           'log_loss': 'neg_log_loss'
           }

def average_score_on_cross_val_classification(clf, X, y, scoring=scoring, optimizer_score = 'accuracy', cv=8):
    """
    Evaluates a given model/estimator using cross-validation
    and returns a dict containing the absolute vlues of the average (mean) scores
    for classification models.

    clf: scikit-learn classification model
    X: features (no labels)
    y: labels
    scoring: a dictionary of scoring metrics
    cv: cross-validation strategy
    """
    # Score metrics on cross-validated dataset
    scores_dict = cross_validate(clf, X, y, scoring=scoring, cv=cv, n_jobs=-1)
    accuracy = {k: v for k, v in scores_dict.items() if k.startswith(f'test_{optimizer_score}')}

    return round(np.mean(accuracy[f'test_{optimizer_score}']), 5)
    # return the average scores for each metric
#    return {metric: round(np.mean(scores), 5) for metric, scores in scores_dict.items()}


numeric_features = [colnames]
numeric_transformer = Pipeline(steps=[
('imputer', SimpleImputer(strategy='median')),
('scaler', StandardScaler())])

preprocessor = ColumnTransformer(
transformers=[
    ('num', numeric_transformer, numeric_features),
])


clf = Pipeline(steps=[('preprocessor', preprocessor),
                  ('classifier', xgb.XGBClassifier())
              ])

def objective(trial):
    params={
        'classifier__C': trial.suggest_int('classifier__C', 0, 20),
        'classifier__max_depth': trial.suggest_int('classifier__max_depth', 0, 6),
        'classifier__min_child_weight': trial.suggest_float('classifier__min_child_weight', 0, 1),
        'classifier__eta': trial.suggest_float('classifier__eta', 0, 1),
        'classifier__subsample': trial.suggest_float('classifier__subsample', 0, 1),
        'classifier__colsample_bytree': trial.suggest_float('classifier__colsample_bytree', 0, 1)
    }
    
    clf.set_params(**params)
    
    return average_score_on_cross_val_classification(clf, X, Y.values.ravel(), cv = 8)
    #return -np.mean(cross_val_score(clf, X, Y.values.ravel(), cv=8))

mlflc = MLflowCallback(
tracking_uri="/home/jupyter/mlruns/",
metric_name='accuracy',
)


#sampler = SkoptSampler()
study = optuna.create_study(study_name='xgboost_metrics', direction='maximize')
study.optimize(objective, n_trials=30, callbacks=[mlflc])
# clf.set_params(**study.best_params)
# clf.fit(X, Y)
