import pytest

def test_trainer_imports():
    from models import XGBoostTrainer
    assert XGBoostTrainer is not None


@pytest.mark.slow
def test_trainer_smoke(tmp_path):
    # Smoke test: initialize trainer and check methods exist
    from config.config_loader import ConfigLoader
    from models import XGBoostTrainer
    cfg = ConfigLoader('config/default_config.yaml')
    trainer = XGBoostTrainer(cfg, horizon=1)
    assert hasattr(trainer, 'load_and_prepare_data')
    assert hasattr(trainer, 'train')
    assert hasattr(trainer, 'evaluate')
