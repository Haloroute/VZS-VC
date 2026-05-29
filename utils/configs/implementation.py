# Configuration class for the training and evaluation process of the VC system
from dataclasses import dataclass

# Configuration for the training process
@dataclass
class TrainConfig:
    device: str = "cuda" # The device to use for training (e.g., "cuda" for GPU or "cpu" for CPU).
    amp_enable: bool = True # Whether to use automatic mixed precision (AMP) during training for faster computation and reduced memory usage.

    n_epochs: int = 500 # The number of epochs to train the model.
    batch_size: int = 16 # The batch size for training and validation.
    lr: float = 2e-4 # The learning rate for the optimizer.
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
    amp_enable: bool = True # Whether to use automatic mixed precision (AMP) during validation for faster computation and reduced memory usage.

    validate_every_n_epochs: int = 10 # The frequency (in epochs) at which to perform validation during training.
    batch_size: int = 16 # The batch size for validation.