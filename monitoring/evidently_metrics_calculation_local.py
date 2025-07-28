import datetime
import time
import random
import logging 
import uuid
import pytz
import pandas as pd
import io
import psycopg
import joblib

from pathlib import Path
from evidently import DataDefinition, Dataset, Report
from evidently.metrics import ValueDrift, DriftedColumnsCount, MissingValueCount

import mlflow
from mlflow.tracking import MlflowClient
from prefect import flow, task
from prefect.deployments import run_deployment

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s]: %(message)s")


"""
Prepare PostgreSQL database and table
Initialize MLFlow tracking URI
Load production model from MLFlow Model Registry
Get unseen data, simulated by getting randomly from the training dataset
Preprocess unseen data
Predict on the unseen data
Get reference data from MLFlow production model run
Calculate drift metrics at 5 random time intervals
"""

SEND_TIMEOUT = 10

CREATE_TABLE_STATEMENT = """
drop table if exists drift_metrics;
create table drift_metrics (
	timestamp timestamp,
	prediction_drift float,
	num_drifted_columns integer,
	share_missing_values float
)
"""

CONNECTION_STRING = "host=localhost port=5432 user=postgres password=example"
CONNECTION_STRING_DB = CONNECTION_STRING + " dbname=test"


def get_num_cat_features(file_path):
	df = pd.read_csv(file_path)
	df.drop(columns=['claim_status'], inplace=True)
	df.drop(columns=['family_history_3', 'employment_type'], inplace=True)
	non_numerical = ['gender', 'marital_status', 'occupation', 'location', 'prev_claim_rejected', 
					'known_health_conditions', 'uk_residence', 'family_history_1', 'family_history_2', 
					'family_history_4', 'family_history_5', 'product_var_1', 'product_var_2', 
					'product_var_3', 'health_status', 'driving_record', 'previous_claim_rate', 
					'education_level', 'income level', 'n_dependents']
	for column in non_numerical:
		df[column] = df[column].astype('category')

	num_features = [x for x in df.columns if df[x].dtype != 'category']

	return num_features, non_numerical


def load_reference_data(file_path):
	df = pd.read_parquet(file_path)
	return df


def preprocess_data(df):
	df.drop(columns=['family_history_3', 'employment_type'], inplace=True)
	non_numerical = ['gender', 'marital_status', 'occupation', 'location', 'prev_claim_rejected', 
					'known_health_conditions', 'uk_residence', 'family_history_1', 'family_history_2', 
					'family_history_4', 'family_history_5', 'product_var_1', 'product_var_2', 
					'product_var_3', 'health_status', 'driving_record', 'previous_claim_rate', 
					'education_level', 'income level', 'n_dependents']
	for column in non_numerical:
		df[column] = df[column].astype('category')
	return df


def create_unseen_data(file_path, random_state):
	df = pd.read_csv(file_path)
	df.drop(columns=['claim_status'], inplace=True)

	df = preprocess_data(df)

	return df.sample(n=1200, random_state=random_state)


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


def load_model(model_id, experiment_id):
    prod_model = f"mlartifacts/{experiment_id}/models/{model_id}/artifacts/"

    print(f"Loading model from {prod_model}...")
    model = mlflow.pyfunc.load_model(prod_model)
    return model


def apply_model_to_data(model, run_id, df):
	df['predicted_claim_status'] = model.predict(df)
	df['model_run_id'] = run_id
	return df

    
def prep_db():
	with psycopg.connect("host=localhost port=5432 user=postgres password=example", autocommit=True) as conn:
		res = conn.execute("SELECT 1 FROM pg_database WHERE datname='test'")
		if len(res.fetchall()) == 0:
			conn.execute("create database test;")
		with psycopg.connect("host=localhost port=5432 dbname=test user=postgres password=example") as conn:
			conn.execute(CREATE_TABLE_STATEMENT)


def calculate_metrics_postgresql(curr, i, unseen_df, reference_data):

	begin = datetime.datetime(2025, 7, 1, 0, 0)

	num_features, cat_features = get_num_cat_features("data/dataset_from_database.csv")
	data_definition = DataDefinition(
		numerical_columns=num_features + ['predicted_claim_status'],
		categorical_columns=cat_features,
	)

	report = Report(metrics = [
		ValueDrift(column='predicted_claim_status'),
		DriftedColumnsCount(),
		MissingValueCount(column='predicted_claim_status'),
	])

	print("Importing unseen data...")
	unseen_df = Dataset.from_pandas(unseen_df, data_definition=data_definition)
	print("Importing unseen data...")
	reference_data = Dataset.from_pandas(reference_data, data_definition=data_definition)

	run = report.run(reference_data=reference_data, current_data=unseen_df)
	result = run.dict()

	prediction_drift = result['metrics'][0]['value']
	num_drifted_columns = result['metrics'][1]['value']['count']
	share_missing_values = result['metrics'][2]['value']['share']

	# If there's drift, retrain the model
	if result['metrics'][1]['value']['share'] >= 0.5:
		print(f"Drift detected, retraining model...")
		# run_deployment(
		# 	name='claim_status_classification_flow/claims_status_classification'
		# )

	curr.execute(
		"insert into drift_metrics(timestamp, prediction_drift, num_drifted_columns, share_missing_values) values (%s, %s, %s, %s)",
		(begin + datetime.timedelta(i), prediction_drift, num_drifted_columns, share_missing_values)
	)


def batch_monitoring_backfill():

	mlflow_tracking_uri = "http://127.0.0.1:5000"
	print("Connecting to mlflow registry server...")
	client = MlflowClient(mlflow_tracking_uri)

	experiment_name = "claims_status"
	model_name = f"{experiment_name}_classifier"
	experiment_id = client.get_experiment_by_name(experiment_name).experiment_id
	print(f"Experiment ID for {experiment_name}: {experiment_id}")

	# Identify and load Production model
	print(f"Getting production model from registry...")
	run_id, model_id = get_prod_model(client, model_name)
	print(f"Loading model with model_id = {model_id}...")
	model = load_model(model_id, experiment_id)

	input_file_path = Path("data/dataset_from_database.csv")
	reference_data = load_reference_data("data/reference.parquet")
	print("Reference data loaded...")
	print(reference_data.columns)
	reference_data = apply_model_to_data(model, run_id, reference_data.drop(columns=['claim_status']))
	print("Reference data scored...")

	prep_db()
	last_send = datetime.datetime.now() - datetime.timedelta(seconds=10)
	with psycopg.connect(CONNECTION_STRING_DB, autocommit=True) as conn:

		for i in range(0, 5):

			# Generate random unseen data
			print(f"Creating randomly generated unseen data number {i}...")
			unseen_df = create_unseen_data(input_file_path, random_state=i)

			# Score unseen data
			print(f"Scoring the data using model with run_id = {run_id}...")
			apply_model_to_data(model, run_id, unseen_df)
			print(f"Scored the data.")

			with conn.cursor() as curr:
				calculate_metrics_postgresql(curr, i, unseen_df, reference_data)

			new_send = datetime.datetime.now()
			seconds_elapsed = (new_send - last_send).total_seconds()
			if seconds_elapsed < SEND_TIMEOUT:
				time.sleep(SEND_TIMEOUT - seconds_elapsed)
			while last_send < new_send:
				last_send = last_send + datetime.timedelta(seconds=10)
			logging.info("data sent")


if __name__ == '__main__':
	batch_monitoring_backfill()