import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class FluorescenceDataset(Dataset):
    def __init__(self, dataset_path, augment_flip=True, augment_jitter=True, normalize=True):
        """
        Dataset for loading fluorescence microscopy sequences.
        Args:
            dataset_path (str): Path to the dataset (npy files or image folders).
            augment_flip (bool): Whether to flip entire sequences.
            augment_jitter (bool): Whether to add jitter to sequences.
            normalize (bool): Normalize inputs to [0, 1].
        """
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"Dataset file not found: {dataset_path}")
        
        self.dataset_path = dataset_path
        self.augment_flip = augment_flip
        self.augment_jitter = augment_jitter
        self.normalize = normalize

        # Load dataset
        self.data = np.load(dataset_path)  # Shape: (num_sequences, seq_len, 1, H, W)

        # Define transforms
        self.transforms = []
        if normalize:
            self.transforms.append(transforms.Normalize((0.5,), (0.5,)))
        self.transforms = transforms.Compose(self.transforms)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sequence = self.data[idx]  # Shape: (seq_len, 1, H, W)

        # Apply augmentations
        if self.augment_flip and np.random.rand() > 0.5:
            sequence = np.flip(sequence, axis=0).copy()
        if self.augment_jitter:
            sequence += np.random.normal(0, 0.01, sequence.shape)

        # Convert to tensor and apply transforms
        sequence = torch.tensor(sequence, dtype=torch.float32)
        sequence = self.transforms(sequence)
        return sequence

def get_dataloader(dataset_path, batch_size, augment_flip=True, augment_jitter=True, normalize=True):
    dataset = FluorescenceDataset(dataset_path, augment_flip, augment_jitter, normalize)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)