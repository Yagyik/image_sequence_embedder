import os
import yaml
import torch
from training.trainer import Trainer
from training.utils import set_seed, ensure_dir
from data.dataloader import get_dataloader
from models.lstm_encoder_decoder import LSTMEncoderDecoder
from logging.tensorboard_logger import TensorBoardLogger

def main(config_path):
    # Load configuration
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    # Set random seed
    set_seed(42)

    # Ensure logging directory exists
    ensure_dir(config['logging']['log_dir'])

    # Initialize TensorBoard logger
    logger = TensorBoardLogger(config['logging']['log_dir']) if config['logging']['tensorboard'] else None

    # Load dataset
    dataloader = get_dataloader(
        dataset_path=config['data']['dataset_path'],
        batch_size=config['training']['batch_size'],
        augment_flip=config['data']['augment_flip'],
        augment_jitter=config['data']['augment_jitter'],
        normalize=config['data']['normalize']
    )

    # Initialize model
    model = LSTMEncoderDecoder(
        input_dim=config['model']['imsize'],
        hidden_dim=config['model']['ndf'],
        latent_dim=config['model']['latent_dim'],
        num_layers=2,
        bidirectional=True,
        teacher_forcing=config['training']['teacher_forcing']
    )

    # Initialize optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=config['training']['learning_rate'])

    # Initialize trainer
    trainer = Trainer(model, dataloader, optimizer, config)

    # Start training
    trainer.train()

    # Close logger
    if logger:
        logger.close()

if __name__ == "__main__":
    main("configs/vae_lstm.yml")