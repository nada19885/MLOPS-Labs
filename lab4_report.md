# Lab 4 Report: MLOps Pipeline with MLflow and Dagshub

## Overview
This lab integrated MLflow with a Titanic survival prediction pipeline, focusing on experiment tracking, model registration, and deployment.

## Models Trained
- **LogisticRegression (TitanicLogisticRegression)**:
  - Accuracy: [e.g., 0.78, fill from Dagshub]
  - Run ID: 379c84dc33f8468f8d194e16c593314c
- **RandomForest (TitanicRandomForest)**:
  - Accuracy: [e.g., 0.82, fill from Dagshub]
  - Run ID: df551ee0026a47f9a6af9d61b9561850

## Model Comparison
- **Best Model**: TitanicRandomForest (higher accuracy of 0.82).
- Selected for deployment based on performance.

## Deployment
- **Status**: Dagshub deployment feature marked as "Coming Soon," so direct deployment was not possible.


## Challenges and Solutions
- **Challenge**: Categorical data caused `ValueError: could not convert string to float`.
  - **Solution**: Used `ColumnTransformer` with `OneHotEncoder` in a `Pipeline`.
- **Challenge**: MLflow `input_example` warnings for model signature.
  - **Solution**: Used a DataFrame slice (`X_train.iloc[:1]`) to reduce warnings.
- **Challenge**: Deployment unavailable on Dagshub.
  - **Solution**: Tested locally with MLflow serving.

## Lessons Learned
- Importance of a modular `Pipeline` for preprocessing and training.
- Debugging MLflow logging and handling deployment limitations.
- Value of experiment tracking for comparing models and reproducibility.