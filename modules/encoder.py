# Audio Encoder Module for VZS-VC
import librosa, torch

import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor


class AudioEncoder(nn.Module):
    """
    Audio Encoder module that natively processes Mel-spectrograms and downsamples them
    to produce continuous audio embeddings with half the temporal resolution.
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
        # Native STFT and Mel-Spectrogram Setup (No torchaudio)
        # ---------------------------------------------------------
        # 1. Create and register the Hann Window for the STFT
        window = torch.hann_window(n_fft)
        self.register_buffer("window", window)

        # 2. Create and register the Mel Filterbank Matrix using Librosa
        mel_basis = librosa.filters.mel(
            sr=sample_rate,
            n_fft=n_fft,
            n_mels=n_mel_bins
        )
        self.register_buffer("mel_basis", torch.from_numpy(mel_basis).float())

        # Feature Extraction Layer 1 (Local Context - Maintains T)
        self.conv_1 = nn.Conv1d(
            in_channels=n_mel_bins,
            out_channels=d_hidden,
            kernel_size=3,
            stride=1,
            padding=1
        )

        # Feature Extraction Layer 2 (Broader Context via Dilation - Maintains T)
        # Formula: padding = dilation * (kernel_size - 1) // 2 to maintain length
        self.conv_2 = nn.Conv1d(
            in_channels=d_hidden,
            out_channels=d_hidden,
            kernel_size=3,
            stride=1,
            padding=2,
            dilation=2
        )

        # Activation Function
        self.activation = nn.SiLU()

        # Downsampling Layer (Halves temporal resolution: T -> T // 2)
        # Formula: L_out = floor((T + 2*1 - 4) / 2) + 1 = T/2
        self.conv_3 = nn.Conv1d(
            in_channels=d_hidden,
            out_channels=d_audio,
            kernel_size=4,
            stride=2,
            padding=1
        )

        # Initialize parameters
        self.init_params()

    def init_params(self, std: float = 0.02):
        """
        Initialize the parameters of the AudioEncoder using truncated normal initialization 
        for weights and zero initialization for biases.
        """
        # Initialize Conv1d layers
        nn.init.trunc_normal_(self.conv_1.weight, mean=0.0, std=std)
        nn.init.zeros_(self.conv_1.bias)

        nn.init.trunc_normal_(self.conv_2.weight, mean=0.0, std=std)
        nn.init.zeros_(self.conv_2.bias)

        nn.init.trunc_normal_(self.conv_3.weight, mean=0.0, std=std)
        nn.init.zeros_(self.conv_3.bias)

    def extract_mel_spectrogram(self, audio: Tensor) -> Tensor:
        """
        Natively computes the log-Mel-spectrogram from raw audio waveforms.

        Args:
            audio: Tensor of shape (N, L)
        
        Returns:
            Tensor of shape (N, n_mel_bins, T)
        """
        stft_complex = torch.stft(
            audio,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.n_fft,
            window=self.window,
            center=False,
            normalized=False,
            return_complex=True
        ) # (N, n_fft//2 + 1, T)

        magnitudes = torch.abs(stft_complex) # (N, n_fft//2 + 1, T)
        
        # Apply Mel Filterbank: (n_mels, n_fft//2 + 1) @ (N, n_fft//2 + 1, T) -> (N, n_mel_bins, T)
        mel_spec = torch.matmul(self.mel_basis, magnitudes) 
        
        log_mel_spec = torch.log(torch.clamp(mel_spec, min=1e-5))
        return log_mel_spec # (N, n_mel_bins, T)

    def forward(
        self,
        input: Tensor
    ) -> Tensor:
        """
        Forward pass for the Audio Encoder Block (Internal processing).

        Args:
            input (Tensor): Input Mel-spectrogram tensor of shape (N, T, n_mel_bins).

        Returns:
            Tensor: Output audio embedding tensor of shape (N, T // 2, D_audio).
        """
        # Step 1: Feature Extraction (Layer 1)
        hidden: Tensor = self.conv_1(input.permute(0, 2, 1)) # (N, T, n_mel_bins) -> (N, n_mel_bins, T) -> (N, d_hidden, T)
        hidden = self.activation(hidden) # (N, d_hidden, T)

        # Step 2: Feature Extraction (Layer 2)
        hidden = self.conv_2(hidden) # (N, d_hidden, T) -> (N, d_hidden, T)
        hidden = self.activation(hidden) # (N, d_hidden, T)

        # Step 3: Downsampling to produce the final audio embedding
        output: Tensor = self.conv_3(hidden) # (N, d_hidden, T) -> (N, d_audio, T // 2)

        return output.permute(0, 2, 1) # (N, d_audio, T // 2) -> (N, T // 2, d_audio)
    
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