from src.pipeline import load_and_preprocess_data, saved_processed_data, train_and_evaluate
import hydra
from omegaconf import DictConfig

@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig):
    print(cfg)
    X_train, X_test, y_train, y_test = load_and_preprocess_data(
        cfg.pipeline.data.raw_data_path, 
        cfg.pipeline.data.test_size, 
        cfg.pipeline.data.random_state
    )
    saved_processed_data(X_train, X_test, y_train, y_test, cfg.pipeline.data.processed_data_path)

    models = { 
        "logistic": cfg.pipeline.model.logistic,
        "random_forest": cfg.pipeline.model.random_forest,
    }

    best_model_name = None
    best_accuracy = 0
    for model_type, model_params in models.items():
        accuracy = train_and_evaluate(
            X_train, X_test, y_train, y_test, model_type, model_params, cfg.pipeline.model.model_path
        )
        print(f"{model_type} Accuracy: {accuracy:.4f}")
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model_name = model_type
    print(f"Best Model: {best_model_name} with accuracy {best_accuracy:.4f}")

if __name__ == "__main__":
    main()