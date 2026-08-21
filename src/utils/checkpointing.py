"""
Checkpoint saving and loading. Saves regularly — cluster jobs can be killed.
"""

import torch
from pathlib import Path


def save_checkpoint(model, optimizer, epoch: int, metrics: dict, path: str):
    """
    Saves model state, optimizer state, epoch, and metrics to disk.
    Args: model, optimizer, epoch (int), metrics (dict), path (str).
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "metrics": metrics,
    }, path)


def load_checkpoint(model, optimizer, path: str) -> dict:
    """
    Loads state dicts into model/optimizer from the specified path.
    Args: model, optimizer, path (str). Returns: full checkpoint dict.
    """
    checkpoint = torch.load(path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint
