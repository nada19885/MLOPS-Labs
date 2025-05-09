import os
import sys
import pickle
import itertools
import yaml
import mlflow
import dagshub
import numpy as np
import pandas as pd
from mlflow.tracking import MlflowClient
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import mlflow.sklearn
from pipeline import load_and_preprocess_data, train_and_evaluate, saved_processed_data, create_pipeline

with open("conf/pipeline/titanic.yaml", "r") as f:
    titanic_config = yaml.safe_load(f)
with open("conf/pipeline/titanic2.yaml", "r") as f:
    titanic2_config = yaml.safe_load(f)

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

logistic_params = {
    "C": [0.1, 1.0, 10.0],
    "max_iter": [200, 500]
}

random_forest_params = {
    "n_estimators": [50, 100, 200],
    "max_depth": [None, 10, 20]
}

def train_pipeline(pipeline_name, config, model_type, param_grid):
    processed_data_path = "data/processed" if pipeline_name == "titanic" else "data/processed_2"

    X_train, X_test, y_train, y_test = load_and_preprocess_data(
        config["data"]["raw_data_path"],
        config["data"]["test_size"],
        config["data"]["random_state"]
    )

    saved_processed_data(X_train, X_test, y_train, y_test, processed_data_path)

    experiment_name = "Titanic_Experiment"
    mlflow.set_experiment(experiment_name)

    param_keys = list(param_grid.keys())
    param_values = list(param_grid.values())
    param_combinations = list(itertools.product(*param_values))

    best_accuracy = 0
    best_run_id = None
    best_model_name = None

    for params in param_combinations:
        param_dict = dict(zip(param_keys, params))
        run_name = f"{pipeline_name}_{model_type}_{'_'.join([f'{k}_{v}' for k, v in param_dict.items()])}"
        with mlflow.start_run(run_name=run_name):
            mlflow.log_params({"data_config": config["data"], "model_config": param_dict})

            pipeline = create_pipeline(model_type, param_dict)
            pipeline.fit(X_train, y_train)
            y_pred = pipeline.predict(X_test)

            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='binary')
            recall = recall_score(y_test, y_pred, average='binary')
            f1 = f1_score(y_test, y_pred, average='binary')

            mlflow.log_metric("accuracy", float(accuracy))
            mlflow.log_metric("precision", float(precision))
            mlflow.log_metric("recall", float(recall))
            mlflow.log_metric("f1_score", float(f1))

            model_name = f"{pipeline_name}_{model_type}"
            mlflow.sklearn.log_model(pipeline, model_name, input_example=X_train.iloc[:0])

            model_path = f"models/{pipeline_name}/{model_type}_{'_'.join([f'{k}_{v}' for k, v in param_dict.items()])}.pkl"
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            with open(model_path, "wb") as f:
                pickle.dump(pipeline, f)

            model_uri = f"runs:/{mlflow.active_run().info.run_id}/{model_name}"
            mlflow.register_model(model_uri, model_name)

            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_run_id = mlflow.active_run().info.run_id
                best_model_name = model_name

    return best_accuracy, best_run_id, best_model_name

def stage_best_model_to_production(model_name, run_id):
    client = MlflowClient()
    client.transition_model_version_stage(
        name=model_name,
        version=1,
        stage='production',
        archive_existing_versions=True
    )
    print(f"Staged model {model_name} from run {run_id} to Production")

if __name__ == "__main__":
    best_lr_accuracy, best_lr_run_id, best_lr_model_name = train_pipeline(
        "titanic", titanic_config, "logistic", logistic_params)

    best_rf_accuracy, best_rf_run_id, best_rf_model_name = train_pipeline(
        "titanic2", titanic2_config, "random_forest", random_forest_params)

    if best_lr_accuracy > best_rf_accuracy:
        best_model_name = best_lr_model_name
        best_run_id = best_lr_run_id
        best_accuracy = best_lr_accuracy
    else:
        best_model_name = best_rf_model_name
        best_run_id = best_rf_run_id
        best_accuracy = best_rf_accuracy

    stage_best_model_to_production(best_model_name, best_run_id)
    print(f"Best model: {best_model_name} with accuracy {best_accuracy} from run {best_run_id}")
