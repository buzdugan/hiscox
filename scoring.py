#!/usr/bin/env python
# coding: utf-8

import os
from pathlib import Path
from datetime import datetime, timedelta

import pandas as pd
import mlflow

from prefect import flow, task


@task(name="create_daily_data", log_prints=True)
def create_daily_data(file_path, output_path):
    df = pd.read_csv(file_path)
    df.drop(columns=['claim_status'], inplace=True)

    df.head(10).to_csv(output_path, index=False)


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


@task(name="load_model", log_prints=True)
def load_model(run_id):
    # prod_model = f's3://mlflow-artifacts-remote-hiscox/1/models/{run_id}/artifacts/'
    prod_model = f'mlartifacts/2/models/{run_id}/artifacts/'
    model = mlflow.pyfunc.load_model(prod_model)
    return model


@task(name="apply_model", log_prints=True)
def apply_model(model, run_id, df, output_path):

    df['predicted_claim_status'] = model.predict(df)
    df['model_run_id'] = run_id
    
    print(f"Saving the predictions to {output_path}...")
    df.to_csv(output_path, index=False)


@flow(name="claim_status_scoring_flow", log_prints=True)
def score_claim_status():

    # RUN_ID = os.getenv('RUN_ID', "m-a2dd0166170844ecab99d852d6ce412d") # model in S3 bucket
    RUN_ID = "m-9eead17988824cac85cb40c965964150"  # model locally downloaded

    input_file_path = Path("data/dataset_from_database.csv")
    output_file_path = Path("data/scored_dataset.csv")

    yesterday = datetime.now() - timedelta(1)
    yesterday_input_file_path = f"{input_file_path[:-4]}_{yesterday.strftime('%Y-%m-%d')}.csv"

    print(f"Reading yesteraday data from {yesterday_input_file_path}...")
    create_daily_data(input_file_path, yesterday_input_file_path)

    print(f"Reading data from {yesterday_input_file_path}...")
    df = read_dataframe(yesterday_input_file_path) 

    print(f"Loading model with run_id: {RUN_ID}...")
    model = load_model(RUN_ID)

    print(f"Scoring the data using model with run_id: {RUN_ID}...")
    apply_model(model, RUN_ID, df, output_file_path)
    print(f"Scored the data.")


if __name__ == "__main__":
    score_claim_status()
