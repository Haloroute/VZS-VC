# VoiceGenerator network
import torch
import torch.nn as nn

from torch import Tensor

from .encoder import AudioEncoder
from .submodules import LogEmbedding, TransformerBlock


class VoiceGenerator(nn.Module):
    """
    VoiceGenerator is a neural network module designed for voice conversion tasks.
    It consists of multiple blocks that process input features and generate converted features.
    """
    def __init__(
        self,
        d_content: int, d_pitch: int, d_amplitude: int, d_timbre: int,
        n_pitch: int, min_pitch: float, max_pitch: float,
        n_amplitude: int, min_amplitude: float, max_amplitude: float,
        d_conv: int, d_model: int, n_heads: int, d_ff: int, n_layers: int, dropout: float = 0.1,
        sample_rate: int = 24000, n_fft: int = 1024, hop_length: int = 240, n_mel_bins: int = 100
    ):
        """
        Initialize the VoiceGenerator module.

        Args:
            d_content (int): The dimensionality of the content embedding (came from VietASR content features). Should be 512.
            d_pitch (int): The dimensionality of the pitch embedding (after logarithmic embedding).
            d_amplitude (int): The dimensionality of the amplitude embedding (after logarithmic embedding).
            d_timbre (int): The dimensionality of the timbre embedding (came from ERes2Net-V2 timbre features). Should be 192.

            n_pitch (int): The number of bins for pitch embedding.
            min_pitch (float): The minimum value for pitch embedding (should be a positive value). Should be around 32.7 (C1 note).
            max_pitch (float): The maximum value for pitch embedding (should be a positive value). Should be around 1244.5 (D#6 note).

            n_amplitude (int): The number of bins for amplitude embedding.
            min_amplitude (float): The minimum value for amplitude embedding (should be a positive value). Should be around 0.01.
            max_amplitude (float): The maximum value for amplitude embedding (should be a positive value). Should be around 0.85.

            d_conv (int): The dimensionality of the convolutional layer.
            d_model (int): The dimensionality of the model (feature dimension).
            n_heads (int): The number of attention heads in each DiT block.
            d_ff (int): The dimensionality of the feed-forward layer in each DiT block.
            n_layers (int): The number of DiT blocks in the generator.
            dropout (float): The dropout rate for regularization. Default is 0.1.

            sample_rate (int): The sample rate of the input audio waveform (should match the sampling rate used for the Mel-spectrogram features in the dataset, which is 24000).
            n_fft (int): Size of the FFT for the Mel-spectrogram.
            hop_length (int): The distance between neighboring sliding window frames (default 240 for 100Hz at 24kHz).
            n_mel_bins (int): Number of Mel frequency bins in the intermediate spectrogram (should match the number of bins used for the Vocoder, which is 100).
        """
        super().__init__()
        # Initialize model parameters
        self.d_content, self.d_pitch, self.d_amplitude, self.d_timbre = d_content, d_pitch, d_amplitude, d_timbre
        self.n_pitch, self.min_pitch, self.max_pitch = n_pitch, min_pitch, max_pitch
        self.n_amplitude, self.min_amplitude, self.max_amplitude = n_amplitude, min_amplitude, max_amplitude
        self.d_conv, self.d_model, self.n_heads, self.d_ff, self.n_layers, self.dropout = d_conv, d_model, n_heads, d_ff, n_layers, dropout
        self.sample_rate, self.n_fft, self.hop_length, self.n_mel_bins = sample_rate, n_fft, hop_length, n_mel_bins

        # Audio Encoder for extracting audio embeddings from Mel-spectrograms
        self.audio_encoder = AudioEncoder(d_model, d_conv, sample_rate, n_fft, hop_length, n_mel_bins)
        self.cpa_mask_token = nn.Parameter(torch.zeros(d_model)) # Learnable mask token for CPA (Content-Pitch-Amplitude) masking
        self.audio_mask_token = nn.Parameter(torch.zeros(d_model)) # Learnable mask token for audio masking

        # Embedding layers for pitch, amplitude features
        self.pitch_embedding = LogEmbedding(n_pitch, d_pitch, min_pitch, max_pitch)
        self.amplitude_embedding = LogEmbedding(n_amplitude, d_amplitude, min_amplitude, max_amplitude)

        # Projection layers for input features
        self.features_projection = nn.Linear(d_content + d_pitch + d_amplitude, d_model)
        self.timbre_projection = nn.Linear(d_timbre, d_model)

        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)
        ])

        # Projection layer for output features
        self.norm = nn.RMSNorm(d_model)
        self.output_projection = nn.Linear(d_model, 2 * n_mel_bins)

        self.init_params()

    def init_params(self, std: float = 0.02):
        """
        Initialize the parameters of the model using various initialization schemes.
        """
        # Initialize the parameters of embedding layers
        self.audio_encoder.init_params(std=std)
        self.pitch_embedding.init_params(std=std)
        self.amplitude_embedding.init_params(std=std)

        # Initialize the mask tokens with truncated normal distribution
        nn.init.trunc_normal_(self.cpa_mask_token, std=std)
        nn.init.trunc_normal_(self.audio_mask_token, std=std)

        # Initialize the parameters of projection layers
        for linear in [self.features_projection, self.timbre_projection, self.output_projection]:
            if isinstance(linear, nn.Linear):
                nn.init.trunc_normal_(linear.weight, std=std)
                if linear.bias is not None:
                    nn.init.zeros_(linear.bias)

        # Initialize the parameters of Transformer blocks
        for i, block in enumerate(self.transformer_blocks):
            block.init_params(std=std/((i + 1) ** 0.5)) # Scale initialization by sqrt of number of layers for stability

    def forward(
        self,
        content: Tensor, pitch: Tensor, amplitude: Tensor, timbre: Tensor, mel: Tensor,
        mask_indices: Tensor, content_length: Tensor, token_length: Tensor
    ) -> Tensor:
        """
        Forward pass for the VoiceGenerator.

        Args:
            content (Tensor): Content features of shape (N, T', D_content). T' ~ T // 2.
            pitch (Tensor): Pitch features of shape (N, T).
            amplitude (Tensor): Amplitude features of shape (N, T).
            timbre (Tensor): Timbre features of shape (N, D_timbre).
            mel (Tensor): Mel-spectrogram features of shape (N, 2T, n_mel_bins).

            mask_indices (Tensor): Indices of positions in the Mel-spectrogram to be masked, shape (N, T). True for positions to be masked, False for positions to be kept.
            content_length (Tensor): Lengths of content sequences for padding, shape (N,).
            token_length (Tensor): Lengths of pitch & amplitude sequences for padding, shape (N,).

        Returns:
            Tensor: Output Mel-spectrogram features of shape (N, 2T, n_mel_bins).
        """
        N, T = mask_indices.shape

        # Step 1: Interpolate content features to match the target length using nearest-neighbor interpolation based on sequence lengths.
        # This allows the model to handle variable-length sequences and learn to align them during training.
        # Create a common destination index grid for the entire batch
        grid: Tensor = torch.arange(T, device=content.device).unsqueeze(0).expand(N, -1) # (N, T)

        def batched_nearest_interpolate(x: Tensor, src_len: Tensor, dst_len: Tensor) -> Tensor:
            """
            Perform batched nearest-neighbor interpolation on 3D sequences using index gathering.

            Args:
                x (Tensor): Input tensor of shape (N, T_src, D).
                src_len (Tensor): Source sequence lengths of shape (N,).
                dst_len (Tensor): Target sequence lengths of shape (N,).

            Returns:
                Tensor: Interpolated tensor of shape (N, T, D).
            """
            # Calculate the nearest mapping ratio: index = floor(grid * (src_len / dst_len))
            ratio: Tensor = src_len.float() / dst_len.float() # (N,)
            src_indices: Tensor = (grid.float() * ratio.unsqueeze(1)).long() # (N, T)

            # Limit indices to prevent out-of-bounds errors in the padding region
            max_idx: Tensor = (src_len.unsqueeze(1) - 1).clamp(min=0) # (N, 1)
            src_indices: Tensor = torch.clamp(src_indices, min=torch.zeros_like(max_idx), max=max_idx) # (N, T)

            src_indices_expanded: Tensor = src_indices.unsqueeze(-1).expand(-1, -1, x.size(-1)) # (N, T, D_content)
            out: Tensor = torch.gather(x, dim=1, index=src_indices_expanded) # (N, T, D_content)

            return out

        # Perform batched nearest-neighbor interpolation for content features
        content_emb: Tensor = batched_nearest_interpolate(content, content_length, token_length) # (N, T, D_content)

        # Step 2: Extract audio embeddings from the input Mel-spectrogram using the audio encoder.
        audio_emb: Tensor = self.audio_encoder(mel) # (N, T, D_model)

        # Apply mask token based on mask_indices
        audio_emb: Tensor = torch.where(mask_indices.unsqueeze(-1).bool(), self.audio_mask_token, audio_emb) # (N, T, D_model)

        # Step 3: Embed pitch and amplitude features using logarithmic embedding.
        pitch_emb: Tensor = self.pitch_embedding(pitch) # (N, T) -> (N, T, D_pitch)
        amplitude_emb: Tensor = self.amplitude_embedding(amplitude) # (N, T) -> (N, T, D_amplitude)

        # Step 3: Concatenate content, pitch, amplitude, and project to d_model dimension.
        features_emb: Tensor = torch.cat([content_emb, pitch_emb, amplitude_emb], dim=-1) # (N, T, D_content + D_pitch + D_amplitude)
        features_emb: Tensor = self.features_projection(features_emb) # (N, T, D_content + D_pitch + D_amplitude) -> (N, T, D_model)

        # Apply mask token to features embedding based on mask_indices for CPA masking
        features_emb: Tensor = torch.where(~mask_indices.unsqueeze(-1).bool(), self.cpa_mask_token, features_emb) # (N, T, D_model)

        # Step 3.5: Add timbre features (which are constant across time) after projecting to d_model dimension.
        timbre_emb: Tensor = self.timbre_projection(timbre).unsqueeze(1) # (N, D_timbre) -> (N, D_model) -> (N, 1, D_model)
        input: Tensor = features_emb + audio_emb + timbre_emb # (N, T, D_model)        

        # Step 4: Pass through Transformer blocks
        for block in self.transformer_blocks:
            input: Tensor = block(
                input=input,
                input_length=token_length
            ) # (N, T, D_model)

        # Step 5: Project the output features back to D_codec dimension.
        output: Tensor = self.norm(input) # (N, T, D_model)
        output: Tensor = self.output_projection(output) # (N, T, D_model) -> (N, T, 2 * n_mel_bins)
        output = output.view(N, 2 * T, self.n_mel_bins) # (N, T, 2 * n_mel_bins) -> (N, 2T, n_mel_bins)

        # Step 6: Copy the unmasked frames from the input Mel-spectrogram to the output to enforce consistency and stabilize training.
        mel_mask_indices: Tensor = mask_indices.repeat_interleave(repeats=2, dim=-1).unsqueeze(-1).bool() # (N, T) -> (N, 2T) -> (N, 2T, 1)
        output = torch.where(mel_mask_indices, output, mel) # (N, 2T, n_mel_bins)

        return output # (N, 2T, n_mel_bins)