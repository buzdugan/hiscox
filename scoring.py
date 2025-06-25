#!/usr/bin/env python
# coding: utf-8

import os
from pathlib import Path

import pandas as pd
import mlflow

from prefect import flow, task


@task(name="read_data", retries=3, retry_delay_seconds=2)
def read_dataframe(file_path):
    dataset_from_database = pd.read_csv(file_path)
    dataset_from_database = dataset_from_database.head(10)
    # dataset_from_database = collect_from_database(f"SELECT * FROM CLAIMS.DS_DATASET")
    
    dataset_from_database.drop(columns=['claim_status'], inplace=True)
    dataset_from_database.drop(columns=['family_history_3', 'employment_type'], inplace=True)
    non_numerical = ['gender', 'marital_status', 'occupation', 'location', 'prev_claim_rejected', 
                    'known_health_conditions', 'uk_residence', 'family_history_1', 'family_history_2', 
                    'family_history_4', 'family_history_5', 'product_var_1', 'product_var_2', 
                    'product_var_3', 'health_status', 'driving_record', 'previous_claim_rate', 
                    'education_level', 'income level', 'n_dependents']
    for column in non_numerical:
        dataset_from_database[column] = dataset_from_database[column].astype('category')

    return dataset_from_database


@task(name="load_model", log_prints=True)
def load_model(run_id):
    # prod_model = f's3://mlflow-artifacts-remote-hiscox/1/models/{run_id}/artifacts/'
    prod_model = f'mlartifacts/2/models/{run_id}/artifacts/'
    model = mlflow.pyfunc.load_model(prod_model)
    return model


@task(name="apply_model", log_prints=True)
def apply_model(model, run_id, df, output_file):

    df['predicted_claim_status'] = model.predict(df)
    df['model_run_id'] = run_id
    
    print(f"Saving the predictions to {output_file}...")
    df.to_csv(output_file, index=False)


@flow(name="claim_status_scoring_flow", log_prints=True)
def score_claim_status():

    # RUN_ID = os.getenv('RUN_ID', "m-a2dd0166170844ecab99d852d6ce412d") # model in S3 bucket
    RUN_ID = "m-9eead17988824cac85cb40c965964150"  # model locally downloaded

    input_file_path = Path("data/dataset_from_database.csv")
    output_file_path = Path("data/scored_dataset.csv")

    print(f"Reading data from {input_file_path}...")
    df = read_dataframe(input_file_path) 

    print(f"Loading model with run_id: {RUN_ID}...")
    model = load_model(RUN_ID)

    print(f"Scoring the data using model with run_id: {RUN_ID}...")
    apply_model(model, RUN_ID, df, output_file_path)
    print(f"Scored the data.")


if __name__ == "__main__":
    score_claim_status()
