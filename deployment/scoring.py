#!/usr/bin/env python
# coding: utf-8

import os
from pathlib import Path
from datetime import datetime, timedelta

import pandas as pd
import mlflow
from mlflow.tracking import MlflowClient
from prefect import flow, task
from prefect_aws import S3Bucket


@task(name="create_daily_data", log_prints=True)
def create_daily_data(file_path, output_path):
    df = pd.read_csv(file_path)
    df.drop(columns=['claim_status'], inplace=True)

    df.sample(n=1200, random_state=42).to_csv(output_path, index=False)


@task(name="read_data", retries=3, retry_delay_seconds=2)
def read_dataframe(file_path):
    df = pd.read_csv(file_path)
    
    df.drop(columns=['family_history_3', 'employment_type'], inplace=True)
    non_numerical = ['gender', 'marital_status', 'occupation', 'location', 'prev_claim_rejected', 
                    'known_health_conditions', 'uk_residence', 'family_history_1', 'family_history_2', 
                    'family_history_4', 'family_history_5', 'product_var_1', 'product_var_2', 
                    'product_var_3', 'health_status', 'driving_record', 'previous_claim_rate', 
                    'education_level', 'income level', 'n_dependents']
    for column in non_numerical:
        df[column] = df[column].astype('category')

    return df


@task(name="get_production_model", log_prints=True)
def get_prod_model(client, model_name):
    # Get all registered models for model name
    reg_models = client.search_registered_models(
        filter_string=f"name='{model_name}'"
    )

    # Get production model run id and model id
    prod_model_run_id = None
    prod_model_model_id = None
    for reg_model in reg_models:
        for model_version in reg_model.latest_versions:
            if model_version.current_stage == 'Production':
                prod_model_run_id = model_version.run_id
                prod_model_model_id = model_version.source.replace('models:/', '') 
                break

    if prod_model_run_id:
        print(f"Production model run_id for {model_name}: {prod_model_run_id}")
        return prod_model_run_id, prod_model_model_id
    else:   
        print(f"No production model found for {model_name}.")


@task(name="load_model", log_prints=True)
def load_model(model_id, experiment_id):
    prod_model = f's3://mlflow-artifacts-remote-hiscox/{experiment_id}/models/{model_id}/artifacts/'
    print(f"Loading model from {prod_model}...")
    model = mlflow.pyfunc.load_model(prod_model)
    return model


@task(name="apply_model", log_prints=True)
def apply_model(model, run_id, df, output_path):

    df['predicted_claim_status'] = model.predict(df)
    df['model_run_id'] = run_id
    
    print(f"Saving the predictions to {output_path}...")
    df.to_parquet(output_path, engine='pyarrow', compression=None, index=False)


@flow(name="claim_status_scoring_flow", log_prints=True)
def score_claim_status():

    print("Loading aws profile...")
    os.environ["AWS_PROFILE"] = "mlops-user"  # AWS profile name
    tracking_server_host = "ec2-54-221-148-124.compute-1.amazonaws.com" # public DNS of the EC2 instance
    mlflow_tracking_uri = f"http://{tracking_server_host}:5000"
    print("Connecting to mlflow registry server...")
    client = MlflowClient(mlflow_tracking_uri)

    experiment_name = "claims_status"
    model_name = f"{experiment_name}_classifier"
    experiment_id = client.get_experiment_by_name(experiment_name).experiment_id
    print(f"Experiment ID for {experiment_name}: {experiment_id}")

    s3_bucket_block = S3Bucket.load("mlops-s3-bucket") # Store training data on S3 bucket
    # Download data from S3 bucket into prefect temp storage
    s3_bucket_block.download_folder_to_path(from_folder="data", to_folder="data") 

    yesterday = datetime.now() - timedelta(1)
    yesterday_str = yesterday.strftime('%Y_%m_%d')

    input_file_path = Path("data/dataset_from_database.csv")
    yesterday_input_file_path = f"{input_file_path.with_suffix('')}_{yesterday_str}.csv"
    output_file_path = f's3://mlflow-artifacts-remote-hiscox/predictions/scored_dataset_{yesterday_str}.parquet'

    print(f"Creating yesterday data from {yesterday_input_file_path}...")
    create_daily_data(input_file_path, yesterday_input_file_path)

    print(f"Reading data from {yesterday_input_file_path}...")
    df = read_dataframe(yesterday_input_file_path) 

    print(f"Getting production model from registry...")
    run_id, model_id = get_prod_model(client, model_name)
        
    print(f"Loading model with model_id = {model_id}...")
    model = load_model(model_id, experiment_id)

    print(f"Scoring the data using model with run_id = {run_id}...")
    apply_model(model, run_id, df, output_file_path)
    print(f"Scored the data.")
    

if __name__ == "__main__":
    score_claim_status()
