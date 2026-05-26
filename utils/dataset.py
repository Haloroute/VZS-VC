# Utilities for dataset handling
import torch

from tensordict import TensorDict
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence

from .configs import TrainConfig, VieNeuTTSPreprocessedDatasetConfig


# Custom collate function to handle batches of dictionaries
def collate_fn(batch: list[dict], config: VieNeuTTSPreprocessedDatasetConfig) -> TensorDict:
    """
    Custom collate function to handle batches of dictionaries, applying optimized padding to variable-length arrays.
    
    Args:
        batch (list[dict]): A list of dictionaries, each containing keys with variable-length arrays.
        config (VieNeuTTSPreprocessedDatasetConfig): Configuration object containing column names and other settings.
        
    Returns:
        TensorDict: A TensorDict containing the collated and padded data.
    """
    # Pad amplitude embeddings
    amplitude_list: list[Tensor] = [minibatch[config.amplitude_column] for minibatch in batch] # List of 1D tensors with variable lengths
    amplitude_length: Tensor = torch.tensor([a.shape[0] for a in amplitude_list], dtype=torch.long) # (N,)
    amplitude_padded: Tensor = pad_sequence(amplitude_list, batch_first=True, padding_value=0.0) # (N, T_amplitude)

    # Pad content embeddings
    content_list: list[Tensor] = [minibatch[config.content_column] for minibatch in batch] # List of 2D tensors with variable lengths
    content_length: Tensor = torch.tensor([c.shape[0] for c in content_list], dtype=torch.long) # (N,)
    content_padded: Tensor = pad_sequence(content_list, batch_first=True, padding_value=0.0) # (N, T_content, D_content)

    # Pad pitch embeddings
    pitch_list: list[Tensor] = [minibatch[config.pitch_column] for minibatch in batch] # List of 1D tensors with variable lengths
    pitch_length: Tensor = torch.tensor([p.shape[0] for p in pitch_list], dtype=torch.long) # (N,)
    pitch_padded: Tensor = pad_sequence(pitch_list, batch_first=True, padding_value=0.0) # (N, T_pitch)

    # # Process timbre embeddings
    # timbre_list: list[Tensor] = [minibatch[config.timbre_column] for minibatch in batch] # List of 2D tensors with fixed (192) lengths
    # timbre_length: Tensor = torch.tensor([t.shape[0] for t in timbre_list], dtype=torch.long) # (N,)
    # timbre_padded: Tensor = pad_sequence(timbre_list, batch_first=True, padding_value=0.0) # (N, D_timbre)

    # Pad pre-VQ embeddings
    pre_vq_list: list[Tensor] = [minibatch[config.pre_vq_column] for minibatch in batch] # List of 2D tensors with variable lengths
    pre_vq_length: Tensor = torch.tensor([v.shape[0] for v in pre_vq_list], dtype=torch.long) # (N,)
    pre_vq_padded: Tensor = pad_sequence(pre_vq_list, batch_first=True, padding_value=0.0) # (N, T, D_codec)

    # Pad acoustic embeddings
    acoustic_list: list[Tensor] = [minibatch[config.acoustic_column] for minibatch in batch] # List of 2D tensors with variable lengths
    acoustic_length: Tensor = torch.tensor([a.shape[0] for a in acoustic_list], dtype=torch.long) # (N,)
    acoustic_padded: Tensor = pad_sequence(acoustic_list, batch_first=True, padding_value=0.0) # (N, T_timbre, D_timbre)

    # Create a TensorDict to hold the collated data
    collated_batch = {
        "target": pre_vq_padded, # (N, T, D_codec)
        "content": content_padded, # (N, T_content, D_content)
        "pitch": pitch_padded, # (N, T_pitch)
        "amplitude": amplitude_padded, # (N, T_amplitude)
        "timbre": acoustic_padded, # (N, T_timbre, D_timbre)

        "target_length": pre_vq_length, # (N,)
        "content_length": content_length, # (N,)
        "pitch_length": pitch_length, # (N,)
        "amplitude_length": amplitude_length, # (N,)
        "timbre_length": acoustic_length # (N,)
    }
    return TensorDict(collated_batch)

# Function to inject training data such as sampled r and t timesteps and conditioning dropout into the batch
def inject_train_data(batch: TensorDict, train_config: TrainConfig) -> TensorDict:
    """
    Injects training data into the batch, such as sampling r and t timesteps and applying conditioning dropout.

    Args:
        batch (TensorDict): The input batch containing the collated data.
        train_config (TrainConfig): Configuration object containing training settings.

    Returns:
        TensorDict: The modified batch with injected training data.
    """
    N = batch.batch_size
    device = batch.device

    # 1. Sample epsilon noise for the entire batch
    epsilon = torch.randn_like(batch['target']) # (N, T, D_codec)

    # 2. Sample r and t timesteps using Logit-Normal distribution parameters from the training configuration
    # Sample r and t from a normal distribution and then apply the logistic function to get values in the range (0, 1)
    mean, std = train_config.rt_sampler
    r_norm = torch.randn(N, device=device) * std + mean # (N,)
    t_norm = torch.randn(N, device=device) * std + mean # (N,)

    # Apply the logistic function to get values in the range (0, 1)
    r = torch.sigmoid(r_norm) # (N,)
    t = torch.sigmoid(t_norm) # (N,)

    # Ensure that r <= t for each sample in the batch by sorting them
    rt_stacked = torch.stack([r, t], dim=1) # (N, 2)
    rt_sorted, _ = torch.sort(rt_stacked, dim=1) # (N, 2) sorted along the second dimension
    r = rt_sorted[:, 0] # Lấy giá trị nhỏ hơn làm start_timestep
    t = rt_sorted[:, 1] # Lấy giá trị lớn hơn làm end_timestep

    # 3. Apply rnt_rate logic to keep r != t for a subset of the batch
    force_equal_mask = torch.rand(N, device=device) >= train_config.rnt_rate
    r = torch.where(force_equal_mask, t, r)

    # 4. Apply conditioning dropout for training regularization
    drop_cond = torch.rand(N, device=device) < train_config.drop_cond_rate # (N,) boolean tensor where True indicates dropping conditioning for that sample

    # 5. Add the sampled timesteps and drop_cond to the batch
    batch['epsilon'] = epsilon
    batch['start_timestep'] = r
    batch['end_timestep'] = t
    batch['drop_cond'] = drop_cond

    return batch

# Function to inject validation data into the batch
def inject_val_data(batch: TensorDict) -> TensorDict:
    """
    Injects validation data into the batch, such as constant r and t timesteps for validation and no conditioning dropout.

    Args:
        batch (TensorDict): The input batch containing the collated data.

    Returns:
        TensorDict: The modified batch with injected validation data.
    """
    N = batch.batch_size
    device = batch.device

    # 1. Sample epsilon noise for the entire batch
    epsilon = torch.randn_like(batch['target']) # (N, T, D_codec)

    # 2. For validation, we can use fixed r and t values (e.g., r=0.0 and t=1.0) to evaluate the model's performance at the start and end of the diffusion process without randomness
    r = torch.zeros(N, device=device) # (N,) start_timestep = 0.0 for all samples
    t = torch.ones(N, device=device) # (N,) end_timestep = 1.0 for all samples

    # 3. Apply no conditioning dropout for validation
    drop_cond = torch.zeros(N, device=device, dtype=torch.bool) # (N,) boolean tensor where True indicates dropping conditioning for that sample

    # 4. Add the sampled timesteps and drop_cond to the batch
    batch['epsilon'] = epsilon
    batch['start_timestep'] = r
    batch['end_timestep'] = t
    batch['drop_cond'] = drop_cond

    return batch