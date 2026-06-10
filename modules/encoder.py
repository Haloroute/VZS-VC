# Audio Encoder Module for VZS-VC
import librosa, torch

import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor


class AudioEncoder(nn.Module):
    """
    Audio Encoder module that natively processes Mel-spectrograms as 2D images
    (C=1) and downsamples them to produce continuous audio embeddings.
    """
    def __init__(
        self,
        d_audio: int, d_hidden: int,
        sample_rate: int = 24000, n_fft: int = 1024, hop_length: int = 240, n_mel_bins: int = 100
    ):
        """
        Initialize the AudioEncoder with specified parameters for the Mel-transform 
        and convolutional layers.

        Args:
            d_audio (int): Dimensionality of the output audio embedding.
            d_hidden (int): Dimensionality of the hidden convolutional layers.
            sample_rate (int): The sample rate of the input audio waveform.
            n_fft (int): Size of the FFT for the Mel-spectrogram.
            hop_length (int): The distance between neighboring sliding window frames.
            n_mel_bins (int): Number of Mel frequency bins in the intermediate spectrogram.
        """
        super().__init__()
        # Constants for the Audio Encoder
        self.n_mel_bins = n_mel_bins
        self.d_audio = d_audio
        self.d_hidden = d_hidden
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length

        # ---------------------------------------------------------
        # Native STFT and Mel-Spectrogram Setup
        # ---------------------------------------------------------
        window = torch.hann_window(n_fft)
        self.register_buffer("window", window)

        mel_basis = librosa.filters.mel(
            sr=sample_rate,
            n_fft=n_fft,
            n_mels=n_mel_bins
        )
        self.register_buffer("mel_basis", torch.from_numpy(mel_basis).float())

        # ---------------------------------------------------------
        # 2D Convolutional Layers (Treated as 1-channel Grayscale Image)
        # ---------------------------------------------------------
        # Layer 1: Local Spatial-Temporal Context
        self.conv_1 = nn.Conv2d(
            in_channels=1,
            out_channels=d_hidden,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1) # Giữ nguyên H (n_mel_bins) và W (T)
        )

        # Layer 2: Dilation trên trục thời gian = 2
        self.conv_2 = nn.Conv2d(
            in_channels=d_hidden,
            out_channels=d_hidden,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 2),
            dilation=(1, 2)
        )

        # Layer 3: Dilation = 4
        self.conv_3 = nn.Conv2d(
            in_channels=d_hidden,
            out_channels=d_hidden * 2,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 4),
            dilation=(1, 4)
        )

        # Layer 4: Dilation = 8
        self.conv_4 = nn.Conv2d(
            in_channels=d_hidden * 2,
            out_channels=d_hidden * 2,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 8),
            dilation=(1, 8)
        )

        # Layer 5: Dilation = 16
        self.conv_5 = nn.Conv2d(
            in_channels=d_hidden * 2,
            out_channels=d_hidden * 4,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 16),
            dilation=(1, 16)
        )

        # Layer 6: Downsampling Layer (T -> T // 2)
        self.conv_6 = nn.Conv2d(
            in_channels=d_hidden * 4,
            out_channels=d_audio,
            kernel_size=(3, 4),
            stride=(1, 2),
            padding=(1, 1)
        )

        self.activation = nn.LeakyReLU(negative_slope=0.1)
        self.init_params()

    def init_params(self, std: float = 0.02):
        conv_layers = [
            self.conv_1, self.conv_2, self.conv_3, 
            self.conv_4, self.conv_5, self.conv_6
        ]

        for layer in conv_layers:
            nn.init.trunc_normal_(layer.weight, mean=0.0, std=std)
            nn.init.zeros_(layer.bias)

    def extract_mel_spectrogram(self, audio: Tensor) -> Tensor:
        stft_complex = torch.stft(
            audio,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.n_fft,
            window=self.window,
            center=False,
            normalized=False,
            return_complex=True
        ) 

        magnitudes = torch.abs(stft_complex) 
        mel_spec = torch.matmul(self.mel_basis, magnitudes) 
        log_mel_spec = torch.log(torch.clamp(mel_spec, min=1e-5))
        return log_mel_spec # (N, n_mel_bins, T)

    def forward(self, input: Tensor) -> Tensor:
        """
        Forward pass for the Audio Encoder Block (Internal processing).

        Args:
            input (Tensor): Input Mel-spectrogram tensor of shape (N, T, n_mel_bins).

        Returns:
            Tensor: Output audio embedding tensor of shape (N, T // 2, D_audio).
        """
        # (N, T, n_mel_bins) -> (N, n_mel_bins, T) -> (N, 1, n_mel_bins, T)
        hidden = input.transpose(1, 2).unsqueeze(1)

        # Feature Extraction
        hidden = self.activation(self.conv_1(hidden)) # (N, d_hidden, n_mel_bins, T)
        hidden = self.activation(self.conv_2(hidden)) # (N, d_hidden, n_mel_bins, T)
        hidden = self.activation(self.conv_3(hidden)) # (N, 2 * d_hidden, n_mel_bins, T)
        hidden = self.activation(self.conv_4(hidden)) # (N, 2 * d_hidden, n_mel_bins, T)
        hidden = self.activation(self.conv_5(hidden)) # (N, 4 * d_hidden, n_mel_bins, T)

        # Max Pooling theo chiều tần số (dim=2 tương ứng với n_mel_bins)
        # (N, 4 * d_hidden, n_mel_bins, T // 2) -> (N, 4 * d_hidden, 1, T // 2)
        hidden = torch.max(hidden, dim=2, keepdim=True)[0]

        # Downsampling theo chiều thời gian T
        output = self.conv_6(hidden) # (N, d_audio, 1, T // 2)

        # (N, d_audio, 1, T // 2) -> (N, T // 2, d_audio)
        return output.squeeze(2).transpose(1, 2)

    def inference(self, x: Tensor) -> Tensor:
        """
        Perform complete inference from raw 24kHz audio waveform to downsampled audio embeddings.

        Args:
            x (Tensor): A 3D tensor containing the audio data in mono, 24kHz format (shape: (N, 1, L)).

        Returns:
            Tensor: The output audio embedding after inference (shape: (N, T // 2, d_audio)).
        """
        with torch.inference_mode():
            # Step 1: Format Audio Waveform to 2D shape (N, L)
            if x.dim() == 3 and x.size(1) == 1:
                x = x.squeeze(1)

            # Step 2: Strict Truncation to prevent Sequence Mismatch
            # Multiply hop_length by 2 (e.g., 240 * 2 = 480) to ensure exact division for the downsampler
            L = x.shape[-1]
            valid_length = (L // (self.hop_length * 2)) * (self.hop_length * 2)
            x_clean = x[:, :valid_length]

            # Step 3: Exact Right-Padding
            pad_amount = self.n_fft - self.hop_length
            left_pad = pad_amount // 2
            right_pad = pad_amount - left_pad
            x_padded = F.pad(x_clean, (left_pad, right_pad), mode='constant', value=0.0) # (N, L_padded)

            # Step 4: Native Mel-Spectrogram Extraction
            mel: Tensor = self.extract_mel_spectrogram(x_padded) # (N, n_mel_bins, T)

            # Step 5: Forward pass through Convolutional network
            audio_emb: Tensor = self.forward(mel.permute(0, 2, 1)) # (N, n_mel_bins, T) -> (N, T, n_mel_bins) -> (N, T // 2, d_audio)

            return audio_emb