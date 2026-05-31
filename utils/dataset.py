# Utilities for dataset handling
import torch

from tensordict import TensorDict
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence

from .configs import VieNeuTTSPreprocessedDatasetConfig


# Function to apply FSQ bounding math
def apply_fsq_bound(z: torch.Tensor, n_bins: int = 4) -> torch.Tensor:
    """Helper function to apply FSQ bounding math."""
    h = (n_bins - 1) * (1 + 1e-3) / 2 # h = 1.5015
    o = 0.5 if n_bins % 2 == 0 else 0.0
    s = torch.tensor(o / h)
    return torch.tanh(z + s.atanh()) * h - o


# Function to map continuous representations to discrete FSQ labels (c) and vice versa, specifically designed for L=4 levels of FSQ
def continuous_to_discrete_label(z: torch.Tensor, n_bins: int = 4, ignore_value: float = -100.0) -> torch.Tensor:
    """
    Maps continuous representations to discrete FSQ labels (c).

    This function applies a non-linear transformation (tanh) and scaling to bound 
    the continuous tensor `z`, then quantizes it to the nearest integer, and 
    shifts the values to non-negative indices (labels). It is specifically 
    designed for Finite Scalar Quantization (FSQ) with L=4 levels.

    Args:
        z (torch.Tensor): Continuous input tensor of arbitrary shape.
        n_bins (int): The number of discrete bins for FSQ. For L=4, this should be set to 4.
        ignore_value (float): The padding value that should be ignored in the target sequences.

    Returns:
        torch.Tensor: A discrete tensor of the same shape as `z`, containing 
            integer labels in the range [0, 3], except for ignored indices (dtype: torch.long).
    """
    # Step 1: Create a mask for ignore_value to ensure that padding values are not mapped to valid labels
    mask_ignore = (z == ignore_value)

    # Step 2: Apply FSQ bounding for residual = first(self.layers).bound(x)
    z_bound_1 = apply_fsq_bound(z, n_bins=n_bins)

    # Step 3: Apply FSQ bounding for round_ste(self.bound(z)) inside FSQ quantizer
    z_bound_2 = apply_fsq_bound(z_bound_1, n_bins=n_bins)

    # Step 4: Quantize to nearest integer
    q = torch.round(z_bound_2)

    # Step 5: Shift to non-negative labels (0, 1, 2, 3)
    c = (q + n_bins // 2).long()
    c = torch.clamp(c, min=0, max=n_bins - 1) # Ensure labels are within [0, n_bins-1]

    # Step 6: Apply the ignore index mask
    c = c.masked_fill(mask_ignore, int(ignore_value))

    return c


# Function to map discrete FSQ labels (c) back to quantized continuous values (q)
def discrete_label_to_continuous(c: torch.Tensor, n_bins: int = 4) -> torch.Tensor:
    """
    Maps discrete FSQ labels (c) back to quantized continuous values (q).

    This function performs the inverse shift operation, converting the integer 
    labels back to the quantized floating-point checkpoints expected by the 
    NeuCodec decoder.

    Args:
        c (torch.Tensor): Discrete input tensor of arbitrary shape, containing 
            integer labels (typically in the range [0, 3]).
        n_bins (int): The number of discrete bins for FSQ. For L=4, this should be set to 4.

    Returns:
        torch.Tensor: A quantized continuous tensor of the same shape as `c` 
            (dtype: torch.float32).
    """
    # Hyperparameters for FSQ with L=4
    h = (n_bins - 1) * (1 + 1e-3) / 2 # h = 1.5015
    o = 0.5 if n_bins % 2 == 0 else 0.0
    s = torch.tensor(o / h)

    # Step 1: Inverse shift operation
    q = c.float() - n_bins // 2

    # Step 2: Inverse bound back to z
    z = ((q + o) / h).atanh() - s.atanh()

    return z


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
        
        # # Đệm lên bội số của 32
        # padded = pad_and_align(t_list, multiple=32, padding_value=padding_value)
        padded = pad_sequence(t_list, batch_first=True, padding_value=padding_value)
        return padded, lengths

    # Trích xuất và đệm toàn bộ các đặc trưng
    amplitude_padded, amplitude_length = process_feature(config.amplitude_column)
    content_padded, content_length = process_feature(config.content_column)
    pitch_padded, pitch_length = process_feature(config.pitch_column)
    acoustic_padded, acoustic_length = process_feature(config.acoustic_column)
    pre_vq_padded, pre_vq_length = process_feature(config.pre_vq_column, padding_value=config.ignore_value)
    pre_vq_padded = continuous_to_discrete_label(pre_vq_padded, ignore_value=config.ignore_value)

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