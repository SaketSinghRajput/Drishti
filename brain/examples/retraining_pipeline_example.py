"""Example demonstrating retraining pipeline usage.

This example is intentionally self-contained and uses dry-run settings.
"""
import os
from config.config_loader import ConfigLoader
from models.retrain_pipeline import RetrainingPipeline

cfg = ConfigLoader().as_dict()
# ensure dry-run deployment
cfg.setdefault('retraining', {}).setdefault('deployment', {})['auto_deploy'] = False
rp = RetrainingPipeline(cfg)

# Example dataset path (CSV)
dataset = 'data/example_retraining.csv'
if not os.path.exists(dataset):
    os.makedirs(os.path.dirname(dataset) or '.', exist_ok=True)
    with open(dataset, 'w') as fh:
        fh.write('feature1,feature2,label\n1,2,0\n3,4,1\n')

outdir = 'data/retraining_example_out'
res = rp.run_full_pipeline(dataset, outdir)
print('Result:', res)
