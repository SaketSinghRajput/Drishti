import os
import tempfile
import json

from brain.models.retrain_pipeline import RetrainingPipeline
from config.config_loader import ConfigLoader


def test_pipeline_basic_validation_runs():
    # Create a minimal config and CSV
    cfg = ConfigLoader().as_dict()
    rp = RetrainingPipeline(cfg)
    # create a tiny CSV
    tmp = tempfile.mkdtemp()
    csv_path = os.path.join(tmp, 'data.csv')
    with open(csv_path, 'w') as fh:
        fh.write('feature1,feature2,label\n1,2,0\n3,4,1\n')
    out = os.path.join(tmp, 'out')
    r = rp.run_validation_only(csv_path)
    assert isinstance(r, dict)


def test_run_full_pipeline_dryrun():
    cfg = ConfigLoader().as_dict()
    # set auto_deploy false to avoid external deployments
    cfg.setdefault('retraining', {}).setdefault('deployment', {})['auto_deploy'] = False
    rp = RetrainingPipeline(cfg)
    tmp = tempfile.mkdtemp()
    csv_path = os.path.join(tmp, 'data.csv')
    with open(csv_path, 'w') as fh:
        fh.write('feature1,feature2,label\n1,2,0\n3,4,1\n')
    out = os.path.join(tmp, 'out')
    r = rp.run_full_pipeline(csv_path, out)
    assert 'stages' in r
    assert os.path.exists(os.path.join(out, 'retraining_report.json'))
