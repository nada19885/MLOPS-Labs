

from src.pipeline import (load_and_preprocess_data, saved_processed_data,
                          train_and_evaluate)


def main():
    data_path = "data/raw/train.csv"
    processed_data_path = "data/processed"
    model_path = "models"
    test_size = 0.2
    random_state = 42

    X_train, X_test, y_train, y_test = load_and_preprocess_data(
        data_path, test_size, random_state
    )
    saved_processed_data(X_train, X_test, y_train, y_test, processed_data_path)

    models = {
        "logistic": {"max_iter": 200},
        "random_forest": {"n_estimators": 100, "random_state": 42},
    }

    best_model_name = None
    best_accuracy = 0
    for model_type, model_params in models.items():
        accuracy = train_and_evaluate(
            X_train, X_test, y_train, y_test, model_type, model_params, model_path
        )
        print(f"{model_type} Accuracy : {accuracy:0.4f}")
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model_name = model_type
    print(f"Best Model: {best_model_name} with accuracy {best_accuracy:.4f}")


if __name__ == "__main__":
    main()
