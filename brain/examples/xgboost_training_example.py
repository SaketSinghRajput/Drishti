"""Example script demonstrating XGBoost training workflow.

This is intended as a runnable example using small synthetic data or a provided
`data/labeled` directory of JSON windows.
"""
from config.config_loader import ConfigLoader
from models import XGBoostTrainer
import os


def main():
    cfg = ConfigLoader('config/default_config.yaml')
    trainer = XGBoostTrainer(cfg, horizon=10)
    data_dir = 'data/labeled'
    if not os.path.exists(data_dir):
        print('No labeled data found in data/labeled — this example expects real data.')
        return
    X_train, X_val, X_test, y_train, y_val, y_test, groups, sample_weights = trainer.load_and_prepare_data(data_dir, use_smote=False, use_weights=True)
    print('Training with', len(X_train), 'samples')
    trainer.train(X_train, y_train, X_val=X_val, y_val=y_val, groups=groups, sample_weights=sample_weights)
    outdir = 'examples/output/h10'
    trainer.save_model(os.path.join(outdir, 'model.json'))
    trainer.evaluate(X_test, y_test, outdir)


if __name__ == '__main__':
    main()
