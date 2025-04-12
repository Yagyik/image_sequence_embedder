import torch
import torch.nn.functional as F

def reconstruction_loss(recon_x, x):
    return F.mse_loss(recon_x, x)

def kl_divergence(mu, logvar):
    return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

def elbo_loss(recon_x, x, mu, logvar, beta=1.0):
    recon_loss = reconstruction_loss(recon_x, x)
    kl_loss = kl_divergence(mu, logvar)
    return recon_loss + beta * kl_loss, recon_loss, kl_loss