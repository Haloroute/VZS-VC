# Utilities for dataset handling
import torch

from tensordict import TensorDict
from torch.nn.utils.rnn import pad_sequence

from .configs import VieNeuTTSPreprocessedDatasetConfig


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
        padded = pad_sequence(t_list, batch_first=True, padding_value=padding_value)
        return padded, lengths

    # Trích xuất và đệm toàn bộ các đặc trưng
    content_padded, content_length = process_feature(config.content_column)
    amplitude_padded, _ = process_feature(config.amplitude_column)
    pitch_padded, pitch_length = process_feature(config.pitch_column)
    timbre_padded, _ = process_feature(config.timbre_column)

    # Xử lý riêng biệt đối với target
    target_in_list, target_out_list = [], []
    start_token = torch.tensor([config.start_token], dtype=torch.long)
    end_token = torch.tensor([config.end_token], dtype=torch.long)
    for minibatch in batch:
        code = minibatch[config.code_column]
        target_in_list.append(torch.cat([start_token, code]))
        target_out_list.append(torch.cat([code, end_token]))

    # Cập nhật chiều dài thực tế của target sau khi thêm start/end token
    target_length = torch.tensor([t.shape[0] for t in target_in_list], dtype=torch.long)

    # Đệm list target_in và target_out
    target_in_padded = pad_sequence(target_in_list, batch_first=True)
    target_out_padded = pad_sequence(target_out_list, batch_first=True, padding_value=config.ignore_value)

    # Đóng gói vào TensorDict
    return TensorDict({
        'content': content_padded,
        'pitch': pitch_padded,
        'amplitude': amplitude_padded,
        'timbre': timbre_padded,
        'target_in': target_in_padded,
        'target_out': target_out_padded,

        'content_length': content_length,
        'source_length': pitch_length,
        'target_length': target_length
    }, batch_size=[N])