import torch
import umap
import matplotlib.pyplot as plt

def visualize_latents(latents, labels=None):
    if not isinstance(latents, torch.Tensor):
        raise ValueError("Latents must be a torch.Tensor")
    reducer = umap.UMAP()
    embeddings = reducer.fit_transform(latents)
    plt.scatter(embeddings[:, 0], embeddings[:, 1], c=labels, cmap='Spectral')
    plt.colorbar()
    plt.show()