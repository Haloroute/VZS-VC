# Utilities for dataset handling
from sympy import content
import librosa, torch
import torch.nn.functional as F

from tensordict import TensorDict
from torch import Tensor
from torchcodec import AudioSamples
from torchcodec.decoders import WavDecoder
from torch.nn.utils.rnn import pad_sequence

from .configs import TrainConfig, VieNeuTTSPreprocessedDatasetConfig, VoiceGeneratorModuleConfig


# Custom collate function to handle batches of dictionaries
def collate_fn(
    batch: list[dict],
    dataset_config: VieNeuTTSPreprocessedDatasetConfig,
    train_config: TrainConfig,
    model_config: VoiceGeneratorModuleConfig
) -> TensorDict:
    """
    Collate function to process a batch of data samples (dictionaries) into a single TensorDict with proper padding and masking.
    """
    N = len(batch)

    def process_feature(column_name: str, padding_value: float = 0.0) -> tuple[Tensor, Tensor]:
        """
        Helper function to extract a specific feature from the batch, pad it, and return the padded tensor along with the original lengths.

        Args:
            column_name (str): The key in the batch dictionaries corresponding to the feature to be processed
            padding_value (float): The value to use for padding shorter sequences (default is 0.0)

        Returns:
            tuple[Tensor, Tensor]: The padded tensor and the original lengths.
        """
        # Trích xuất list tensor từ batch
        t_list = [minibatch[column_name] for minibatch in batch]

        # Lưu lại chiều dài thực tế để sinh mask phục vụ tính Loss sau này
        lengths = torch.tensor([t.shape[0] for t in t_list], dtype=torch.long)

        # Đệm list tensor
        padded = pad_sequence(t_list, batch_first=True, padding_value=padding_value)
        return padded, lengths

    # Trích xuất và đệm toàn bộ các đặc trưng
    content_padded, content_lengths = process_feature(dataset_config.content_column)
    amplitude_padded, _ = process_feature(dataset_config.amplitude_column)
    pitch_padded, _ = process_feature(dataset_config.pitch_column)

    # Khởi tạo tham số cho việc xử lý audio (Mel-spectrogram và masking)
    window = torch.hann_window(model_config.n_fft)
    mel_basis = librosa.filters.mel(
        sr=24000,
        n_fft=model_config.n_fft,
        n_mels=model_config.n_mel_bins
    )
    mel_basis_tensor = torch.from_numpy(mel_basis).float()

    # Xử lý riêng đối với audio
    mel_list, mask_indices_list, mel_lengths = [], [], []
    for minibatch in batch:
        # Lấy audio từ batch
        audio_samples: AudioSamples = WavDecoder(minibatch[dataset_config.audio_column]['bytes']).get_all_samples()
        audio: Tensor = audio_samples.data.squeeze(0)  # (1, L) -> (L,)
        
        # 1. Truncation: Đảm bảo độ dài chia hết hoàn toàn cho hop_length * 2
        L = audio.shape[0]
        valid_length = (L // (model_config.hop_length * 2)) * (model_config.hop_length * 2)
        audio_clean = audio[:valid_length]

        # 2. Padding: Tính toán padding chính xác cho center=False
        pad_amount = model_config.n_fft - model_config.hop_length
        left_pad = pad_amount // 2
        right_pad = pad_amount - left_pad
        audio_padded = F.pad(audio_clean, (left_pad, right_pad), mode='constant', value=0.0) # (L_padded,)

        # 3. Native STFT Extraction
        stft_complex = torch.stft(
            audio_padded,
            n_fft=model_config.n_fft,
            hop_length=model_config.hop_length,
            win_length=model_config.n_fft,
            window=window,
            center=False,
            normalized=False,
            return_complex=True
        ) # Shape: (n_fft//2 + 1, 2T)

        # 4. Magnitude to Mel
        magnitudes = torch.abs(stft_complex) # (n_fft//2 + 1, 2T)
        mel = torch.matmul(mel_basis_tensor, magnitudes) # (n_mel_bins, n_fft//2 + 1) @ (n_fft//2 + 1, 2T) -> (n_mel_bins, 2T)
        mel = torch.log(torch.clamp(mel, min=1e-5))

        # Đưa về shape (2T, n_mel_bins) để đưa vào pad_sequence
        mel = mel.transpose(0, 1)

        # Tính T tương ứng với 50Hz để sinh mask
        double_T = mel.shape[0]
        T = double_T // 2

        # Sample min_mask_ratio <= mask_ratio <= max_mask_ratio
        mask_ratio = torch.rand(1).item() * (train_config.mask_ratio[1] - train_config.mask_ratio[0]) + train_config.mask_ratio[0]
        mask_len = max(int(T * mask_ratio), 1)

        start_idx = torch.randint(0, T - mask_len + 1, (1,)).item() if T >= mask_len else 0
        mask_indices = torch.zeros(T, dtype=torch.bool) # (T,) Chú ý: mask sinh trên không gian T (50Hz)
        mask_indices[start_idx:start_idx + mask_len] = True

        # Append results to the lists
        mel_list.append(mel) # (2T, n_mel_bins)
        mask_indices_list.append(mask_indices) # (T,)
        mel_lengths.append(T) # Lưu chiều dài T để sinh mask đúng sau này

    # Đệm list token, mask_indices và target
    mel_padded = pad_sequence(mel_list, batch_first=True, padding_value=0) # (N, 2T, n_mel_bins)
    mask_indices_padded = pad_sequence(mask_indices_list, batch_first=True, padding_value=False) # (N, T)
    mel_lengths = torch.tensor(mel_lengths, dtype=torch.long) # (N,)

    # Đóng gói vào TensorDict
    return TensorDict({
        "content": content_padded, # (N, T', D_content)
        "pitch": pitch_padded, # (N, T)
        "amplitude": amplitude_padded, # (N, T)
        "mel": mel_padded, # (N, 2T, n_mel_bins)

        "mask_indices": mask_indices_padded, # (N, T)
        "content_length": content_lengths, # (N,)
        "token_length": mel_lengths, # (N,)
    }, batch_size=[N])