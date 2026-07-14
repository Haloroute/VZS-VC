# Configuration class for the training and evaluation process of the VC system
import torch
from dataclasses import dataclass

# Configuration for the training process
@dataclass
class TrainConfig:
    device: str = "cuda" # The device to use for training (e.g., "cuda" for GPU or "cpu" for CPU).
    compiled: bool = True # Whether to use torch.compile for potential speed improvements during training (optional, can be disabled if it causes issues).
    amp: torch.dtype = torch.float16 # The automatic mixed precision (AMP) mode to use during training (available options: torch.float16, torch.bfloat16, torch.float32).
    n_workers: int = 4 # The number of worker processes to use for data loading during training

    n_epochs: int = 100 # The number of epochs to train the model.
    batch_size: int = 32 # The batch size for training and validation.
    lr: float = 5e-4 # The learning rate for the optimizer.
    beta: tuple[float, float] = (0.9, 0.9) # The beta parameters for the AdamW optimizer.
    weight_decay: float = 0.01 # The weight decay for regularization.
    clip_grad_norm: float = 1.0 # The maximum norm for gradient clipping to prevent exploding gradients.
    ema_decay: float = 0.999 # The decay rate for the Exponential Moving Average (EMA) of model parameters.
    start_factor: float = 0.1 # The initial learning rate factor for the learning rate scheduler (relative to the base learning rate).
    n_warmup_epochs: int = 10 # The number of epochs for the learning rate warmup phase.

    mask_ratio: tuple[float, float] = (0.1, 0.7) # Range for random mask ratio during training
    lambda_recon: float = 1.0 # Weight for the reconstruction loss component
    lambda_adv: float = 0.05 # Weight for the adversarial loss component
    lambda_fm: float = 0.1 # Weight for the feature matching loss component

    checkpoint_folder: str = "checkpoints" # The folder where model checkpoints will be saved during training.
    save_every_n_epochs: int = 1 # The frequency (in epochs) at which to save model checkpoints during training.
    seed: int = 42 # The random seed for reproducibility of results across runs.

# Configuration for the validation process
@dataclass
class ValidationConfig:
    device: str = "cuda" # The device to use for validation (e.g., "cuda" for GPU or "cpu" for CPU).
    compiled: bool = True # Whether to use torch.compile for potential speed improvements during validation (optional, can be disabled if it causes issues).
    amp: torch.dtype = torch.float16 # The automatic mixed precision (AMP) mode to use during validation (available options: torch.float16, torch.bfloat16, torch.float32).
    n_workers: int = 4 # The number of worker processes to use for data loading during validation

    validate_every_n_epochs: int = 1 # The frequency (in epochs) at which to perform validation during training.
    batch_size: int = 32 # The batch size for validation.

# Configuration for the inference process
@dataclass
class InferenceConfig:
    device: str = "cuda" # The device to use for inference (e.g., "cuda" for GPU or "cpu" for CPU).
    compiled: bool = False # Whether to use torch.compile for potential speed improvements during inference (optional, can be disabled if it causes issues).
    amp: torch.dtype = torch.float32 # The automatic mixed precision (AMP) mode to use during inference (available options: torch.float16, torch.bfloat16, torch.float32).

# Configuration for real-time voice conversion (VC) inference
@dataclass
class RealTimeConfig:
    device: str = "cuda" # The device to use for real-time inference (e.g., "cuda" for GPU or "cpu" for CPU).
    compiled: bool = False # Whether to use torch.compile for potential speed improvements during real-time inference (optional, can be disabled if it causes issues).
    amp: torch.dtype = torch.float32 # The automatic mixed precision (AMP) mode to use during real-time inference (available options: torch.float16, torch.bfloat16, torch.float32).

    sample_rate: int = 24000 # The sampling rate for the input and output audio (should match the sampling rate used for training, which is 24000).
    n_channels: int = 1 # The number of audio channels (1 for mono, 2 for stereo). For real-time VC, mono is typically used to minimize latency and complexity.
    chunk_size_ms: int = 2000 # The size of each audio chunk in milliseconds for processing (e.g., 1000ms corresponds to 24000 samples at 24kHz).
    overlap_size_ms: int = 160 # The size of the overlap region in milliseconds for cross-fading between chunks (e.g., 160ms corresponds to 3840 samples at 24kHz).
    n_max_chunks: int = 10 # The maximum number of chunks to keep in the queue