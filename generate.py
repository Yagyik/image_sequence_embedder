import torch

def generate(model, n_samples, latent_dim):
    z = torch.randn(n_samples, latent_dim)
    generated_sequences = model.decode(z)
    return generated_sequences