## Lab 5 Report: Hyperparameter Tuning and Multi-Metric Evaluation

### Overview
Enhanced the Titanic pipeline to run multiple trials in a single experiment, track additional metrics, and stage the best model to production.

### Experiment Details
- **Experiment Name**: Titanic_Experiment
- **Trials**:
  - LogisticRegression: 6 runs (C=[0.1, 1.0, 10.0], max_iter=[200, 500])
  - RandomForest: 9 runs (n_estimators=[50, 100, 200], max_depth=[None, 10, 20])
- **Metrics Tracked**: Accuracy, Precision, Recall, F1-Score

### Best Model
- **Model**: titanic2_random_forest (n_estimators=50, max_depth=None)
- **Run ID**: 91ed742426fb499c85910430ad693928
- **Accuracy**: 0.813
- **Staged to Production**: Successfully tagged as "Production" in the MLflow Model Registry.

### Challenges
- **Challenge**: `input_example` warnings in MLflow.
  - **Solution**: Ensured `input_example` contained data using `X_train.iloc[:1].copy()`.
- **Challenge**: Deprecated `transition_model_version_stage` method.
  - **Solution**: Noted for future update to use tags.