import os
import tempfile
from config.config_loader import ConfigLoader
from inference.explanation_logger import ExplanationLogger


def test_explanation_logger_writes_jsonl(tmp_path):
    cfg = ConfigLoader('config/default_config.yaml')
    outdir = tmp_path / "expl"
    outdir.mkdir()
    logger = ExplanationLogger(cfg, output_dir=str(outdir))
    decision = {"risk_score": 0.5, "alert_level": "low", "alert_active": True, "uncertainty_flag": False, "metadata": {}}
    explanation = {"message": "test", "top_features": []}
    features = {"a": 1}
    sensor_health = {}
    logger.log_prediction(decision, explanation, features, sensor_health)
    logger.close()
    files = list(outdir.glob('explanations_*.jsonl'))
    assert len(files) == 1
    # ensure file is non-empty
    assert files[0].stat().st_size > 0
