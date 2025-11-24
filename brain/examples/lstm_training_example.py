"""Example script showing LSTM training workflow.
"""
from config.config_loader import ConfigLoader
from models.train_lstm import LSTMTrainer


def main():
    cfg = ConfigLoader('config/default_config.yaml')
    trainer = LSTMTrainer(cfg, horizon=10)
    X_train, y_train, X_val, y_val, X_test, y_test = trainer.load_and_prepare_data('data/labeled')
    trainer.build_model(input_shape=X_train.shape[1:])
    trainer.train(X_train, y_train, X_val, y_val, output_dir='examples/output/lstm_h10')
    trainer.evaluate(X_test, y_test, output_dir='examples/output/lstm_h10')


if __name__ == '__main__':
    main()
