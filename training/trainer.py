import torch
from logging.tensorboard_logger import TensorBoardLogger
from training.losses import elbo_loss

class Trainer:
    def __init__(self, model, dataloader, optimizer, config):
        self.model = model
        self.dataloader = dataloader
        self.optimizer = optimizer
        self.config = config
        self.logger = TensorBoardLogger(config['logging']['log_dir']) if config['logging']['tensorboard'] else None

    def train(self):
        for epoch in range(self.config['training']['max_epochs']):
            for batch in self.dataloader:
                self.optimizer.zero_grad()
                recon_x, mu, logvar = self.model(batch)
                loss, recon_loss, kl_loss = elbo_loss(recon_x, batch, mu, logvar, beta=1.0)
                loss.backward()
                self.optimizer.step()

                # Log losses
                if self.logger and epoch % self.config['logging']['log_frequency'] == 0:
                    self.logger.log_scalar('Loss/Total', loss.item(), epoch)
                    self.logger.log_scalar('Loss/Reconstruction', recon_loss.item(), epoch)
                    self.logger.log_scalar('Loss/KL', kl_loss.item(), epoch)

        # Close logger
        if self.logger:
            self.logger.close()