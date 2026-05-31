# Utilities for logging training progress, validation results, and other relevant information during the training process.
import torch, warnings
import torch.nn as nn

from torch.amp import GradScaler
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.optim.swa_utils import AveragedModel

from modules import VoiceGenerator


# Function to unwrap torch.compile for clean checkpoint saving
def get_raw_model(m: nn.Module) -> nn.Module:
    """Unwraps torch.compile for clean checkpoint saving."""
    if hasattr(m, "_orig_mod"):
        m = m._orig_mod
    if hasattr(m, "module"):
        m = m.module
    return m


# Function to save a checkpoint of the model, optimizer, and training state
def save_checkpoint(
    checkpoint_path: str,
    model: VoiceGenerator | None,
    averaged_model: AveragedModel | None,
    optimizer: Optimizer | None,
    scheduler: LRScheduler | None,
    scaler: GradScaler | None,
    epoch: int,
    loss: float | None = None,
    accuracy: float | None = None
):
    """
    Save a checkpoint of the model, optimizer, and training state.

    Args:
        checkpoint_path (str): The path where the checkpoint will be saved.
        model (VoiceGenerator | None): The model to save.
        averaged_model (AveragedModel | None): The averaged model to save.
        optimizer (Optimizer | None): The optimizer to save.
        scheduler (LRScheduler | None): The learning rate scheduler to save.
        scaler (GradScaler | None): The gradient scaler to save.
        epoch (int): The current epoch number.
        loss (float | None): The current training loss. Optional, defaults to None.
        accuracy (float | None): The current training accuracy. Optional, defaults to None.
    """
    checkpoint = {
        'model_state_dict': get_raw_model(model).state_dict() if model is not None else None,
        'averaged_model_state_dict': get_raw_model(averaged_model).state_dict() if averaged_model is not None else None,
        'optimizer_state_dict': optimizer.state_dict() if optimizer is not None else None,
        'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None,
        'scaler_state_dict': scaler.state_dict() if scaler is not None else None,
        'epoch': epoch,
        'loss': loss,
        'accuracy': accuracy
    }
    torch.save(checkpoint, checkpoint_path)


# Function to load a checkpoint and restore the model, optimizer, and training state
def load_checkpoint(
    checkpoint_path: str,
    model: VoiceGenerator | None,
    averaged_model: AveragedModel | None,
    optimizer: Optimizer | None,
    scheduler: LRScheduler | None,
    scaler: GradScaler | None,
) -> tuple[dict, int, int | None, float | None]:
    """
    Load a checkpoint and restore the model, optimizer, and training state.

    Args:
        model (VoiceGenerator | None): The model to restore.
        averaged_model (AveragedModel | None): The averaged model to restore.
        optimizer (Optimizer | None): The optimizer to restore.
        scheduler (LRScheduler | None): The learning rate scheduler to restore.
        scaler (GradScaler | None): The gradient scaler to restore.
        checkpoint_path (str): The path to the checkpoint file.

    Returns:
        tuple: A tuple containing the checkpoint dictionary, epoch number, loss and accuracy.
    """
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    # Load model state dicts with warnings if keys are missing
    if model is not None and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    else:
        warnings.warn("Model state dict not found in checkpoint. Model weights not loaded.")
    
    # Load averaged model state dict with warning if keys are missing
    if averaged_model is not None and 'averaged_model_state_dict' in checkpoint:
        averaged_model.load_state_dict(checkpoint['averaged_model_state_dict'], strict=False)
    else:
        warnings.warn("Averaged model state dict not found in checkpoint. Averaged model weights not loaded.")

    # Load optimizers with warnings if keys are missing
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    else:
        warnings.warn("Optimizer state dict not found in checkpoint. Optimizer state not loaded.")

    # Load scheduler with warning if keys are missing
    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    else:
        warnings.warn("Scheduler state dict not found in checkpoint. Scheduler state not loaded.")

    # Load gradient scaler state dict with warning if keys are missing
    if scaler is not None and 'scaler_state_dict' in checkpoint:
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
    else:
        warnings.warn("Gradient scaler state dict not found in checkpoint. Gradient scaler state not loaded.")

    epoch, loss, accuracy = checkpoint.get('epoch'), checkpoint.get('loss'), checkpoint.get('accuracy')
    return checkpoint, epoch, loss, accuracy