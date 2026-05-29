# Utilities for logging training progress, validation results, and other relevant information during the training process.
import torch

from torch import Tensor
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.optim.swa_utils import AveragedModel
from torchdata.stateful_dataloader import StatefulDataLoader

from modules import MeanFlowsGenerator


# Function to save a checkpoint of the model, optimizer, and training state
def save_checkpoint(
    checkpoint_path: str,
    model: MeanFlowsGenerator,
    averaged_model: AveragedModel,
    optimizer: Optimizer,
    scheduler: LRScheduler,
    train_loader: StatefulDataLoader,
    epoch: int,
    step: int = None,
    loss: float = None
):
    """
    Save a checkpoint of the model, optimizer, and training state.

    Args:
        checkpoint_path (str): The path where the checkpoint will be saved.
        model (MeanFlowsGenerator): The model to save.
        averaged_model (AveragedModel): The averaged model to save.
        optimizer (Optimizer): The optimizer to save.
        scheduler (LRScheduler): The learning rate scheduler to save.
        train_loader (StatefulDataLoader): The training data loader to save the state of.
        epoch (int): The current epoch number.
        step (int): The current training step number. Optional, defaults to None.
        loss (float): The current training loss. Optional, defaults to None.
    """
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'averaged_model_state_dict': averaged_model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'train_loader_state_dict': train_loader.state_dict(),
        'epoch': epoch,
        'step': step,
        'loss': loss
    }
    torch.save(checkpoint, checkpoint_path)


# Function to load a checkpoint and restore the model, optimizer, and training state
def load_checkpoint(
    checkpoint_path: str,
    model: MeanFlowsGenerator,
    averaged_model: AveragedModel,
    optimizer: Optimizer,
    scheduler: LRScheduler,
    train_loader: StatefulDataLoader,
) -> tuple[dict, int, int | None, float | None]:
    """
    Load a checkpoint and restore the model, optimizer, and training state.

    Args:
        model (MeanFlowsGenerator): The model to restore.
        averaged_model (AveragedModel): The averaged model to restore.
        optimizer (Optimizer): The optimizer to restore.
        scheduler (LRScheduler): The learning rate scheduler to restore.
        train_loader (StatefulDataLoader): The training data loader to restore the state of.
        checkpoint_path (str): The path to the checkpoint file.

    Returns:
        tuple: A tuple containing the checkpoint dictionary, epoch number, step number and loss.
    """
    checkpoint = torch.load(checkpoint_path)

    model.load_state_dict(checkpoint['model_state_dict'])
    averaged_model.load_state_dict(checkpoint['averaged_model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    train_loader.load_state_dict(checkpoint['train_loader_state_dict'])

    epoch, step, loss = checkpoint.get('epoch'), checkpoint.get('step'), checkpoint.get('loss')
    return checkpoint, epoch, step, loss