# Utilities for logging training progress, validation results, and other relevant information during the training process.
import torch, warnings
import torch.nn as nn

from torch.amp import GradScaler
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.optim.swa_utils import AveragedModel

from modules import VoiceDiscriminator, VoiceGenerator


# Function to unwrap torch.compile for clean checkpoint saving
def get_raw_model(m: nn.Module) -> nn.Module:
    """Unwraps torch.compile for clean checkpoint saving."""
    if hasattr(m, "module"):
        m = m.module
    if hasattr(m, "_orig_mod"):
        m = m._orig_mod
    return m


# Function to save a checkpoint of the model, optimizer, and training state
def save_checkpoint(
    checkpoint_path: str,
    model: VoiceGenerator | None,
    averaged_model: AveragedModel | None,
    discriminator: VoiceDiscriminator | None,

    generator_optimizer: Optimizer | None,
    discriminator_optimizer: Optimizer | None,
    generator_scheduler: LRScheduler | None,
    discriminator_scheduler: LRScheduler | None,
    generator_scaler: GradScaler | None,
    discriminator_scaler: GradScaler | None,

    epoch: int
):
    """
    Save a checkpoint of the model, optimizer, and training state.

    Args:
        checkpoint_path (str): The path where the checkpoint will be saved.
        model (VoiceGenerator | None): The model to save.
        averaged_model (AveragedModel | None): The averaged model to save.
        discriminator (VoiceDiscriminator | None): The discriminator to save.

        generator_optimizer (Optimizer | None): The generator optimizer to save.
        discriminator_optimizer (Optimizer | None): The discriminator optimizer to save.
        generator_scheduler (LRScheduler | None): The generator learning rate scheduler to save.
        discriminator_scheduler (LRScheduler | None): The discriminator learning rate scheduler to save.
        generator_scaler (GradScaler | None): The generator gradient scaler to save.
        discriminator_scaler (GradScaler | None): The discriminator gradient scaler to save.

        epoch (int): The current epoch number.
    """
    checkpoint = {
        'model_state_dict': get_raw_model(model).state_dict() if model is not None else None,
        'averaged_model_state_dict': get_raw_model(averaged_model).state_dict() if averaged_model is not None else None,
        'discriminator_state_dict': get_raw_model(discriminator).state_dict() if discriminator is not None else None,

        'generator_optimizer_state_dict': generator_optimizer.state_dict() if generator_optimizer is not None else None,
        'discriminator_optimizer_state_dict': discriminator_optimizer.state_dict() if discriminator_optimizer is not None else None,
        'generator_scheduler_state_dict': generator_scheduler.state_dict() if generator_scheduler is not None else None,
        'discriminator_scheduler_state_dict': discriminator_scheduler.state_dict() if discriminator_scheduler is not None else None,
        'generator_scaler_state_dict': generator_scaler.state_dict() if generator_scaler is not None else None,
        'discriminator_scaler_state_dict': discriminator_scaler.state_dict() if discriminator_scaler is not None else None,

        'epoch': epoch
    }
    torch.save(checkpoint, checkpoint_path)


# Function to load a checkpoint and restore the model, optimizer, and training state
def load_checkpoint(
    checkpoint_path: str,
    model: VoiceGenerator | None,
    averaged_model: AveragedModel | None,
    discriminator: VoiceDiscriminator | None,

    generator_optimizer: Optimizer | None,
    discriminator_optimizer: Optimizer | None,
    generator_scheduler: LRScheduler | None,
    discriminator_scheduler: LRScheduler | None,
    generator_scaler: GradScaler | None,
    discriminator_scaler: GradScaler | None,
) -> tuple[dict, int]:
    """
    Load a checkpoint and restore the model, optimizer, and training state.

    Args:
        checkpoint_path (str): The path to the checkpoint file.
        model (VoiceGenerator | None): The model to restore.
        averaged_model (AveragedModel | None): The averaged model to restore.
        discriminator (VoiceDiscriminator | None): The discriminator to restore.

        generator_optimizer (Optimizer | None): The generator optimizer to restore.
        discriminator_optimizer (Optimizer | None): The discriminator optimizer to restore.
        generator_scheduler (LRScheduler | None): The generator learning rate scheduler to restore.
        discriminator_scheduler (LRScheduler | None): The discriminator learning rate scheduler to restore.
        generator_scaler (GradScaler | None): The generator gradient scaler to restore.
        discriminator_scaler (GradScaler | None): The discriminator gradient scaler to restore.

    Returns:
        tuple: A tuple containing the checkpoint dictionary and epoch number.
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

    # Load discriminator state dict with warning if keys are missing
    if discriminator is not None and 'discriminator_state_dict' in checkpoint:
        discriminator.load_state_dict(checkpoint['discriminator_state_dict'], strict=False)
    else:
        warnings.warn("Discriminator state dict not found in checkpoint. Discriminator weights not loaded.")

    # Load optimizers with warnings if keys are missing
    if generator_optimizer is not None and 'generator_optimizer_state_dict' in checkpoint:
        generator_optimizer.load_state_dict(checkpoint['generator_optimizer_state_dict'])
    else:
        warnings.warn("Generator optimizer state dict not found in checkpoint. Generator optimizer state not loaded.")

    if discriminator_optimizer is not None and 'discriminator_optimizer_state_dict' in checkpoint:
        discriminator_optimizer.load_state_dict(checkpoint['discriminator_optimizer_state_dict'])
    else:
        warnings.warn("Discriminator optimizer state dict not found in checkpoint. Discriminator optimizer state not loaded.")

    # Load schedulers with warnings if keys are missing
    if generator_scheduler is not None and 'generator_scheduler_state_dict' in checkpoint:
        generator_scheduler.load_state_dict(checkpoint['generator_scheduler_state_dict'])
    else:
        warnings.warn("Generator scheduler state dict not found in checkpoint. Generator scheduler state not loaded.")

    if discriminator_scheduler is not None and 'discriminator_scheduler_state_dict' in checkpoint:
        discriminator_scheduler.load_state_dict(checkpoint['discriminator_scheduler_state_dict'])
    else:
        warnings.warn("Discriminator scheduler state dict not found in checkpoint. Discriminator scheduler state not loaded.")

    # Load gradient scaler state dicts with warning if keys are missing
    if generator_scaler is not None and 'generator_scaler_state_dict' in checkpoint:
        generator_scaler.load_state_dict(checkpoint['generator_scaler_state_dict'])
    else:
        warnings.warn("Generator gradient scaler state dict not found in checkpoint. Generator gradient scaler state not loaded.")

    if discriminator_scaler is not None and 'discriminator_scaler_state_dict' in checkpoint:
        discriminator_scaler.load_state_dict(checkpoint['discriminator_scaler_state_dict'])
    else:
        warnings.warn("Discriminator gradient scaler state dict not found in checkpoint. Discriminator gradient scaler state not loaded.")

    epoch = checkpoint.get('epoch')
    return checkpoint, epoch