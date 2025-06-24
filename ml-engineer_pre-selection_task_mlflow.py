#!/usr/bin/env python
# coding: utf-8

import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from scipy import stats
from sklearn.datasets import *
from sklearn.metrics import cohen_kappa_score, accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split, RandomizedSearchCV

import mlflow
from mlflow.tracking import MlflowClient
from prefect import flow, task


@task(name="mlflow_initialization")
def init_mlflow(mlflow_tracking_uri, mlflow_experiment_name):
    client = MlflowClient(mlflow_tracking_uri)

    mlflow.set_tracking_uri(mlflow_tracking_uri)

    try:
        experiment_id = mlflow.create_experiment(mlflow_experiment_name)
    except:
        experiment_id = mlflow.get_experiment_by_name(
            mlflow_experiment_name
        ).experiment_id
    mlflow.set_experiment(experiment_id=experiment_id)

    return client


@task(name="read_data", retries=3, retry_delay_seconds=2)
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


@task(name="split_data")
def create_train_test_datasets(dataset_from_database):
    target = 'claim_status'
    X, y = dataset_from_database.drop(target, axis=1), dataset_from_database[[target]]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1889)

    return X_train, X_test, y_train, y_test


@task(name="hyperparameter_tuning", log_prints=True)
def hyperparameter_tuning(X_train, y_train, eval_set, eval_metrics):
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
    
    return parameter_gridSearch.best_params_


@task(name="train_model", log_prints=True)
def train_model(X_train, y_train, X_test, y_test, artifact_path):
    with mlflow.start_run() as run:
        mlflow.set_tag("model", "xgboost")

        # Build the evaluation set & metric list
        eval_set = [(X_train, y_train)]
        eval_metrics = ['auc', 'rmse', 'logloss']

        best_params = hyperparameter_tuning(X_train, y_train, eval_set, eval_metrics)
        mlflow.log_params(best_params)

        # Fit model with the best parameters
        model = xgb.XGBClassifier(
            objective='binary:logistic',
            eval_metric=eval_metrics,
            early_stopping_rounds=15,
            enable_categorical=True,
            **best_params
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
        mlflow.xgboost.log_model(model, artifact_path=artifact_path)

        return run.info.run_id


@task(name="register_model", log_prints=True)
def register_model(run_id, model_name, artifact_path):    
    mlflow.register_model(
        model_uri=f"runs:/{run_id}/{artifact_path}",
        name=model_name
    )


@task(name="productionize_model", log_prints=True)
def stage_model(client, run_id, model_name):
    # Get all registered models for model name
    reg_models = client.search_registered_models(
        filter_string=f"name='{model_name}'"
    )

    # Get trained model version and production model run id
    prod_model_run_id = None
    for reg_model in reg_models:
        for model_version in reg_model.latest_versions:
            if model_version.run_id == run_id:
                trained_model_version = model_version.version

            if model_version.current_stage == 'Production':
                prod_model_run_id = model_version.run_id

    # If no model in production, promote the trained model to production
    if not prod_model_run_id:
        client.transition_model_version_stage(
                name=model_name,
                version=trained_model_version,
                stage="Production",
                archive_existing_versions=True,
            )
        print(f'Productionized version {trained_model_version} of {model_name} model.')
    else:
        # Get the metrics for production and trained models
        prod_model_run = client.get_run(prod_model_run_id)
        prod_model_kappa = prod_model_run.data.metrics['kappa_test']
        trained_model_run = client.get_run(run_id)
        trained_model_kappa = trained_model_run.data.metrics['kappa_test']

        # If trained model's kappa score better than production model, promote to production else archive
        if trained_model_kappa > prod_model_kappa:
            client.transition_model_version_stage(
                name=model_name,
                version=trained_model_version,
                stage="Production",
                archive_existing_versions=True,
            )
            print(f'Productionized version {trained_model_version} of {model_name} model.')
        else:
            client.transition_model_version_stage(
                name=model_name,
                version=trained_model_version,
                stage="Archived",
            )
            print(f'Archived version {trained_model_version} of {model_name} model.')
    

@flow(name="claim_status_classification_flow")
def main_flow():
    np.random.seed(1889)

    print("Loading aws profile...")
    # os.environ["AWS_PROFILE"] = "mlops-user"  # AWS profile name
    # tracking_server_host = "ec2-13-218-161-214.compute-1.amazonaws.com" # public DNS of the EC2 instance
    # mlflow_tracking_uri = f"http://{tracking_server_host}:5000"
    mlflow_tracking_uri = "http://127.0.0.1:5000" # run locally

    experiment_name = "claims_status"
    model_name = f"{experiment_name}_classifier"
    artifact_path = "models_mlflow"

    print("Connecting to mlflow tracking server...")
    client = init_mlflow(mlflow_tracking_uri, experiment_name)
    print("Connected to mlflow tracking server...")

    dataset_from_database = read_dataframe()
    X_train, X_test, y_train, y_test = create_train_test_datasets(dataset_from_database)
    print("Model training starting...")
    run_id = train_model(X_train, y_train, X_test, y_test, artifact_path)
    
    register_model(run_id, model_name, artifact_path)
    print(f"Registered model {model_name} with run_id: {run_id}.")

    stage_model(client, run_id, model_name)
    print(f"Staged model {model_name} with run_id: {run_id}.")
    

if __name__ == "__main__":
    main_flow()
