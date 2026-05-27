# Utilities for dataset handling
import torch

from tensordict import TensorDict
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence

from .configs import TrainConfig, VieNeuTTSPreprocessedDatasetConfig


# Function to pad a list of tensors to the same length and then further pad the time dimension to a multiple of 'multiple'
def pad_and_align(tensor_list: list[Tensor], multiple: int = 32, padding_value: float = 0.0) -> Tensor:
    """
    Đệm danh sách các tensor có độ dài khác nhau thành một batch, 
    sau đó tiếp tục đệm chiều thời gian (chiều 1) lên bội số của 'multiple'.
    """
    # 1. Đệm các tensor đến chiều dài lớn nhất hiện có trong batch
    padded: Tensor = pad_sequence(tensor_list, batch_first=True, padding_value=padding_value)
    
    # 2. Tính toán chiều dài đệm mục tiêu
    T = padded.shape[1]
    target_len = ((T + multiple - 1) // multiple) * multiple
    
    if target_len == T:
        return padded
        
    # 3. Tạo tensor đệm để bù đắp phần thiếu hụt và nối (concatenate) vào chiều thời gian
    pad_amount = target_len - T
    pad_shape = list(padded.shape)
    pad_shape[1] = pad_amount  # Chỉ thay đổi kích thước của chiều thời gian
    
    padding_tensor = torch.full(
        pad_shape, 
        padding_value, 
        dtype=padded.dtype, 
        device=padded.device
    )
    
    return torch.cat([padded, padding_tensor], dim=1)


# Custom collate function to handle batches of dictionaries
def collate_fn(batch: list[dict], config: VieNeuTTSPreprocessedDatasetConfig) -> TensorDict:
    """
    Hàm collate tùy chỉnh xử lý đệm dữ liệu đầu vào lên bội số của 32.
    """
    N = len(batch)

    def process_feature(column_name: str, padding_value: float = 0.0):
        # Trích xuất list tensor từ batch
        t_list = [minibatch[column_name] for minibatch in batch]
        
        # Lưu lại chiều dài thực tế để sinh mask phục vụ tính Loss sau này
        lengths = torch.tensor([t.shape[0] for t in t_list], dtype=torch.long)
        
        # Đệm lên bội số của 32
        padded = pad_and_align(t_list, multiple=32, padding_value=padding_value)
        return padded, lengths

    # Trích xuất và đệm toàn bộ các đặc trưng
    amplitude_padded, amplitude_length = process_feature(config.amplitude_column)
    content_padded, content_length = process_feature(config.content_column)
    pitch_padded, pitch_length = process_feature(config.pitch_column)
    acoustic_padded, acoustic_length = process_feature(config.acoustic_column)
    pre_vq_padded, pre_vq_length = process_feature(config.pre_vq_column)

    # Đóng gói vào TensorDict
    return TensorDict({
        'amplitude': amplitude_padded,
        'amplitude_length': amplitude_length,
        'content': content_padded,
        'content_length': content_length,
        'pitch': pitch_padded,
        'pitch_length': pitch_length,
        'timbre': acoustic_padded,
        'timbre_length': acoustic_length,
        'target': pre_vq_padded,
        'target_length': pre_vq_length
    }, batch_size=[N])


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
    r = rt_sorted[:, 0] # Lấy giá trị nhỏ hơn làm start_time
    t = rt_sorted[:, 1] # Lấy giá trị lớn hơn làm end_time

    # 3. Apply rnt_rate logic to keep r != t for a subset of the batch
    force_equal_mask = torch.rand(N, device=device) >= train_config.rnt_rate
    r = torch.where(force_equal_mask, t, r)

    # 4. Apply conditioning dropout for training regularization
    drop_cond = torch.rand(N, device=device) < train_config.drop_cond_rate # (N,) boolean tensor where True indicates dropping conditioning for that sample

    # 5. Add the sampled timesteps and drop_cond to the batch
    batch['epsilon'] = epsilon
    batch['start_time'] = r
    batch['end_time'] = t
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
    r = torch.zeros(N, device=device) # (N,) start_time = 0.0 for all samples
    t = torch.ones(N, device=device) # (N,) end_time = 1.0 for all samples

    # 3. Apply no conditioning dropout for validation
    drop_cond = torch.zeros(N, device=device, dtype=torch.bool) # (N,) boolean tensor where True indicates dropping conditioning for that sample

    # 4. Add the sampled timesteps and drop_cond to the batch
    batch['epsilon'] = epsilon
    batch['start_time'] = r
    batch['end_time'] = t
    batch['drop_cond'] = drop_cond

    return batch