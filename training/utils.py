import os
import random
import numpy as np
import torch

def set_seed(seed):
    """
    Set the random seed for reproducibility.
    Args:
        seed (int): The seed value to set.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def save_checkpoint(model, optimizer, epoch, filepath):
    """
    Save a model checkpoint.
    Args:
        model (torch.nn.Module): The model to save.
        optimizer (torch.optim.Optimizer): The optimizer to save.
        epoch (int): The current epoch number.
        filepath (str): The path to save the checkpoint.
    """
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
    }
    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved to {filepath}")

def load_checkpoint(filepath, model, optimizer=None):
    """
    Load a model checkpoint.
    Args:
        filepath (str): The path to the checkpoint file.
        model (torch.nn.Module): The model to load the state into.
        optimizer (torch.optim.Optimizer, optional): The optimizer to load the state into.
    Returns:
        int: The epoch number from the checkpoint.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Checkpoint file not found: {filepath}")
    
    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    print(f"Checkpoint loaded from {filepath}")
    return checkpoint['epoch']

def count_parameters(model):
    """
    Count the number of trainable parameters in a model.
    Args:
        model (torch.nn.Module): The model to count parameters for.
    Returns:
        int: The number of trainable parameters.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def ensure_dir(directory):
    """
    Ensure that a directory exists. If it doesn't, create it.
    Args:
        directory (str): The directory path.
    """
    if not os.path.exists(directory):
        os.makedirs(directory)