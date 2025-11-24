"""Example script showing multimodal fusion training workflow.
"""
from config.config_loader import ConfigLoader
from models.train_multimodal import MultimodalTrainer


def main():
    cfg = ConfigLoader('config/default_config.yaml')
    trainer = MultimodalTrainer(cfg, horizon=10)
    train_t, val_t, test_t = trainer.load_and_prepare_data('data/labeled')
    trainer.build_model(camera_shape=train_t[0].shape[1:], sensor_shape=train_t[1].shape[1:], tabular_shape=(train_t[2].shape[1],))
    trainer.train(train_t, val_t, output_dir='examples/output/multimodal_h10')
    trainer.evaluate(test_t, output_dir='examples/output/multimodal_h10')


if __name__ == '__main__':
    main()
