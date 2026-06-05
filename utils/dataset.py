# Utilities for dataset handling
import torch

from tensordict import TensorDict
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence

from .configs import VieNeuTTSPreprocessedDatasetConfig, VoiceGeneratorModuleConfig


# Function to decode token codes from 1D (T,) to multi-dimensional (T, D_fsq) based on n_bins
def decode_fsq_code(code: torch.Tensor, n_bins: int, d_fsq: int) -> torch.Tensor:
    """
    Decompose token codes from 1D (T,) to multi-dimensional (T, D_fsq) based on n_bins.
    For example with n_bins=4: 100 -> [0, 1, 2, 1, 0, 0, 0, 0]
    """
    powers = torch.pow(n_bins, torch.arange(d_fsq, device=code.device))
    return (code.unsqueeze(-1) // powers) % n_bins


# Custom collate function to handle batches of dictionaries
def collate_fn(
    batch: list[dict],
    dataset_config: VieNeuTTSPreprocessedDatasetConfig,
    model_config: VoiceGeneratorModuleConfig
) -> TensorDict:
    """
    Hàm collate tùy chỉnh xử lý đệm dữ liệu đầu vào lên bội số của 32.
    """
    N = len(batch)

    def process_feature(column_name: str, padding_value: float = 0.0):
        # Trích xuất list tensor từ batch
        t_list = [minibatch[column_name] for minibatch in batch]

        # Lưu lại chiều dài thực tế để sinh mask phục vụ tính Loss sau này
        lengths = torch.tensor([t.shape[0] for t in t_list], dtype=torch.long)

        # Đệm list tensor
        padded = pad_sequence(t_list, batch_first=True, padding_value=padding_value)
        return padded, lengths

    # Trích xuất và đệm toàn bộ các đặc trưng
    content_padded, content_length = process_feature(dataset_config.content_column)
    amplitude_padded, _ = process_feature(dataset_config.amplitude_column)
    pitch_padded, _ = process_feature(dataset_config.pitch_column)
    timbre_padded, _ = process_feature(dataset_config.timbre_column)

    # Xử lý riêng biệt đối với token và target
    token_list, mask_indices_list, target_list, token_length = [], [], [], []

    for minibatch in batch:
        code: Tensor = minibatch[dataset_config.code_column] # (T,)
        T = code.shape[0]

        # Sample min_mask_ration <= mask_ratio <= max_mask_ration
        mask_ratio = torch.rand(1).item() * (dataset_config.max_mask_ration - dataset_config.min_mask_ration) + dataset_config.min_mask_ration
        mask_len = max(int(T * mask_ratio), 1)

        start_idx = torch.randint(0, T - mask_len + 1, (1,)).item() if T >= mask_len else 0
        mask_indices = torch.zeros(T, dtype=torch.bool) # (T,)
        mask_indices[start_idx:start_idx + mask_len] = True

        target = code.clone() # (T,)
        target = decode_fsq_code(target, model_config.n_bins, model_config.d_fsq) # (T, D_fsq)
        target[~mask_indices.unsqueeze(-1).expand(-1, model_config.d_fsq)] = dataset_config.ignore_token # Chỉ giữ lại target cho phần bị mask

        token_list.append(code)
        mask_indices_list.append(mask_indices)
        target_list.append(target)
        token_length.append(T)

    # Chuyển token_length sang tensor
    token_length = torch.tensor(token_length, dtype=torch.long) # (N,)

    # Đệm list token, mask_indices và target
    token_padded = pad_sequence(token_list, batch_first=True, padding_value=0) # (N, T)
    mask_indices_padded = pad_sequence(mask_indices_list, batch_first=True, padding_value=False) # (N, T)
    target_padded = pad_sequence(target_list, batch_first=True, padding_value=dataset_config.ignore_token) # (N, T, D_fsq)

    # Đóng gói vào TensorDict
    return TensorDict({
        'content': content_padded, # (N, T', D_content)
        'pitch': pitch_padded, # (N, T)
        'amplitude': amplitude_padded, # (N, T)
        'timbre': timbre_padded, # (N, D_timbre)
        'token': token_padded, # (N, T)

        'mask_indices': mask_indices_padded, # (N, T)
        'content_length': content_length, # (N,)
        'token_length': token_length, # (N,)
        'target': target_padded, # (N, T, D_fsq)
    }, batch_size=[N])