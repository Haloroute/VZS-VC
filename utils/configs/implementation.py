# Configuration class for the training and evaluation process of the VC system
import torch
from dataclasses import dataclass

# Configuration for the training process
@dataclass
class TrainConfig:
    device: str = "cuda" # The device to use for training (e.g., "cuda" for GPU or "cpu" for CPU).
    compiled: bool = True # Whether to use torch.compile for potential speed improvements during training (optional, can be disabled if it causes issues).
    amp: torch.dtype = torch.float16 # The automatic mixed precision (AMP) mode to use during training (available options: torch.float16, torch.bfloat16, torch.float32).
    n_workers: int = 2 # The number of worker processes to use for data loading during training

    n_epochs: int = 100 # The number of epochs to train the model.
    batch_size: int = 24 # The batch size for training and validation.
    lr: float = 2.5e-4 # The learning rate for the optimizer.
    beta: tuple[float, float] = (0.9, 0.95) # The beta parameters for the AdamW optimizer.
    weight_decay: float = 0.0 # The weight decay for regularization.
    ema_decay: float = 0.9999 # The decay rate for the Exponential Moving Average (EMA) of model parameters.
    start_factor: float = 0.05 # The initial learning rate factor for the learning rate scheduler (relative to the base learning rate).
    n_warmup_epochs: int = 20 # The number of epochs for the learning rate warmup phase.

    checkpoint_folder: str = "checkpoints" # The folder where model checkpoints will be saved during training.
    save_every_n_epochs: int = 1 # The frequency (in epochs) at which to save model checkpoints during training.
    seed: int = 42 # The random seed for reproducibility of results across runs.

# Configuration for the validation process
@dataclass
class ValidationConfig:
    device: str = "cuda" # The device to use for validation (e.g., "cuda" for GPU or "cpu" for CPU).
    compiled: bool = False # Whether to use torch.compile for potential speed improvements during validation (optional, can be disabled if it causes issues).
    amp: torch.dtype = torch.float16 # The automatic mixed precision (AMP) mode to use during validation (available options: torch.float16, torch.bfloat16, torch.float32).
    n_workers: int = 2 # The number of worker processes to use for data loading during validation

    validate_every_n_epochs: int = 1 # The frequency (in epochs) at which to perform validation during training.
    batch_size: int = 24 # The batch size for validation.