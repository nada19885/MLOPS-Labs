# test_config.py
import hydra
from omegaconf import DictConfig

@hydra.main(config_path="conf", config_name="config", version_base=None)
def test(cfg: DictConfig):
    print(cfg)              # Show entire config
    print(cfg.data)         # This should work
    print(cfg.model)        # This should also work

if __name__ == "__main__":
    test()
