#!/usr/bin/env python
# coding: utf-8

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from scipy import stats
from sklearn.datasets import *
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import cohen_kappa_score, accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, roc_auc_score, confusion_matrix, log_loss, roc_curve
from sklearn.model_selection import train_test_split, RandomizedSearchCV

import mlflow

np.random.seed(1889)


mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("claims_status")

models_folder = Path('models')
models_folder.mkdir(exist_ok=True)



# Assumption: We train model daily with data up to and including the previous day 
# Date = date of application
def read_dataframe():
    dataset_from_database = pd.read_csv("dataset_from_database.csv")
    # dataset_from_database = collect_from_database(f"SELECT * FROM CLAIMS.DS_DATASET")
    
    dataset_from_database.drop(columns=['family_history_3', 'employment_type'], inplace=True)
    non_numerical = ['gender', 'marital_status', 'occupation', 'location', 'prev_claim_rejected', 
                    'known_health_conditions', 'uk_residence', 'family_history_1', 'family_history_2', 
                    'family_history_4', 'family_history_5', 'product_var_1', 'product_var_2', 
                    'product_var_3', 'health_status', 'driving_record', 'previous_claim_rate', 
                    'education_level', 'income level', 'n_dependents']
    for column in non_numerical:
        dataset_from_database[column] = dataset_from_database[column].astype('category')

    return dataset_from_database


def train_model(X_train, y_train, X_test, y_test):
    with mlflow.start_run() as run:
        mlflow.set_tag("model", "xgboost")
        # Build the evaluation set & metric list
        eval_set = [(X_train, y_train)]
        eval_metrics = ['auc', 'rmse', 'logloss']

        # Randomized search for hyperparameter tuning
        parameter_gridSearch = RandomizedSearchCV(
            estimator=xgb.XGBClassifier(
            objective='binary:logistic',
            eval_metric=eval_metrics,
            early_stopping_rounds=15,
            enable_categorical=True,
            ),

            param_distributions={
            'n_estimators': stats.randint(50, 500),
            'learning_rate': stats.uniform(0.01, 0.75),
            'subsample': stats.uniform(0.25, 0.75),
            'max_depth': stats.randint(1, 8),
            'colsample_bytree': stats.uniform(0.1, 0.75),
            'min_child_weight': [1, 3, 5, 7, 9],
            },

            cv=5,
            n_iter=5,
            verbose=False,
            scoring='roc_auc',
        )

        parameter_gridSearch.fit(X_train, y_train, eval_set=eval_set, verbose=False)
        mlflow.log_params(parameter_gridSearch.best_params_)

        # Fit model with the best parameters
        model = xgb.XGBClassifier(
            objective='binary:logistic',
            eval_metric=eval_metrics,
            early_stopping_rounds=15,
            enable_categorical=True,
            **parameter_gridSearch.best_params_
            )

        model.fit(X_train, y_train, eval_set=eval_set, verbose=False)

        with open('models/xgboost.bin', 'wb') as f_out:
            pickle.dump(model, f_out)

        # Model evaluation
        train_class_preds = model.predict(X_train)
        test_class_preds = model.predict(X_test)
        train_prob_preds = model.predict_proba(X_train)[:, 1]
        test_prob_preds = model.predict_proba(X_test)[:, 1]

        # Log metrics
        def kappa_score(y_true, y_class_preds):
            y = np.array(y_true)
            y = y.astype(int)
            yhat = np.array(y_class_preds)
            yhat = np.clip(np.round(yhat), np.min(y), np.max(y)).astype(int)
            kappa = round(cohen_kappa_score(yhat, y, weights='quadratic'), 2)
            return kappa
        
        metrics = {
            "kappa_train": kappa_score(y_train, train_class_preds),
            "kappa_test": kappa_score(y_test, test_class_preds),
            "train_accuracy": accuracy_score(y_train, train_class_preds),
            "test_accuracy": accuracy_score(y_test, test_class_preds),
            "train_roc": roc_auc_score(y_train, train_prob_preds),
            "test_roc": roc_auc_score(y_test, test_prob_preds),
            "test_f1": f1_score(y_test, test_class_preds),
            "test_precision": precision_score(y_test, test_class_preds),
            "test_recall": recall_score(y_test, test_class_preds)
        }
        for metric_name, metric_value in metrics.items():
            mlflow.log_metric(metric_name, metric_value)

        # Log the model
        mlflow.xgboost.log_model(model, artifact_path="models_mlflow")

        return run.info.run_id


def run():
    dataset_from_database = read_dataframe()

    target = 'claim_status'
    X, y = dataset_from_database.drop(target, axis=1), dataset_from_database[[target]]
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1889)

    run_id = train_model(X_train, y_train, X_test, y_test)
    print(f"MLflow run_id: {run_id}")
    return run_id
    

if __name__ == "__main__":

    run_id = run()

    with open("run_id.txt", "w") as f:
        f.write(run_id)