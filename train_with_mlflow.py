import os
import sys

from mlflow.models import model_config

from pipeline import load_and_preprocess_data, train_and_evaluate, saved_processed_data, create_pipeline
import mlflow
import pickle
import dagshub
import numpy as np
import pandas as pd
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import mlflow.sklearn

# Load configs
with open("conf/pipeline/titanic.yaml" , "r") as f:
    titanic_config = yaml.safe_load(f)
with open("conf/pipeline/titanic2.yaml" , "r") as f:
    titanic2_config = yaml.safe_load(f)

# Dagshub Auth
dagshub_token = os.getenv("DAGSHUB_TOKEN")
dagshub_username = os.getenv("DAGSHUB_USERNAME")
if not dagshub_username or not dagshub_token:
    raise ValueError("DAGSHUB_USERNAME and/or DAGSHUB_TOKEN environment variables are not set.")

dagshub.auth.add_app_token(token=dagshub_token)
dagshub.init(
    repo_owner=dagshub_username,
    repo_name="MLOps-ITI",
    mlflow=True
)

mlflow.set_tracking_uri("https://dagshub.com/excellence.nadamaher/Mlops-ITI.mlflow")

def train_pipeline(pipeline_name):
    if pipeline_name == "titanic":
        config = titanic_config
        model_type = "logistic"
        model_params = config["model"]["logistic"]
        processed_data_path = "data/processed"
    elif pipeline_name == "titanic2":
        config = titanic2_config
        model_type = "random_forest"
        config = titanic2_config
        model_params = config["model"]["random_forest"]
        processed_data_path = "data/processed_2"
    else:
        raise ValueError("Unsupported pipeline")

    X_train, X_test, y_train, y_test = load_and_preprocess_data(
        config["data"]["raw_data_path"],
        config["data"]["test_size"],
        config["data"]["random_state"]
    )

    saved_processed_data(X_train, X_test, y_train, y_test, processed_data_path)
    with mlflow.start_run(run_name=f"{pipeline_name}_run"):
        mlflow.log_params({"data_config": config["data"], "model_config": model_params})
        pipeline = create_pipeline(model_type, model_params)
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        mlflow.log_metric("accuracy", float(accuracy))
        mlflow.sklearn.log_model(
            pipeline,
            f"{pipeline_name}_model",
            input_example=X_train.iloc[:1].values.tolist()
        )
        model_path = f"models/{pipeline_name}/{model_type}.pkl"
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        with open(model_path, "wb") as f:
            pickle.dump(pipeline, f)

    return accuracy

if __name__ == "__main__":
    train_pipeline("titanic")
    train_pipeline("titanic2")
