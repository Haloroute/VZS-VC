# MeanFlows network
import torch

import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor
from .submodules import LogEmbedding, TransformerBlock


class VoiceGenerator(nn.Module):
    """
    VoiceGenerator is a neural network module designed for voice conversion tasks.
    It consists of multiple blocks that process input features and generate converted features based on the given time step.
    """
    def __init__(
        self,
        d_content: int, d_pitch: int, d_amplitude: int, d_timbre: int, d_codec: int, n_bins: int,
        n_pitch: int, min_pitch: float, max_pitch: float, n_amplitude: int, min_amplitude: float, max_amplitude: float,
        d_model: int, n_heads: int, d_ff: int, n_layers: int, dropout: float = 0.1
    ):
        """
        Initialize the VoiceGenerator module.

        Args:
            d_content (int): The dimensionality of the content embedding (came from VietASR content features). Should be 512.
            d_pitch (int): The dimensionality of the pitch embedding (after logarithmic embedding).
            d_amplitude (int): The dimensionality of the amplitude embedding (after logarithmic embedding).
            d_timbre (int): The dimensionality of the timbre embedding (came from NeuCodec acoustic features). Should be 1024.
            d_codec (int): The dimensionality of the codec embedding (used for Finite Scalar Quantization). Should be 8.
            n_bins (int): The number of bins for each dimension (used for Finite Scalar Quantization). Should be 4.

            n_pitch (int): The number of bins for pitch embedding.
            min_pitch (float): The minimum value for pitch embedding (should be a positive value). Should be around 32.7 (C1 note).
            max_pitch (float): The maximum value for pitch embedding (should be a positive value). Should be around 1244.5 (D#6 note).
            n_amplitude (int): The number of bins for amplitude embedding.
            min_amplitude (float): The minimum value for amplitude embedding (should be a positive value). Should be around 0.01.
            max_amplitude (float): The maximum value for amplitude embedding (should be a positive value). Should be around 0.85.

            d_model (int): The dimensionality of the model (feature dimension).
            n_heads (int): The number of attention heads in each DiT block.
            d_ff (int): The dimensionality of the feed-forward layer in each DiT block.
            n_layers (int): The number of DiT blocks in the generator.
            dropout (float): The dropout rate for regularization. Default is 0.1.
        """
        super().__init__()
        # Initialize model parameters
        self.d_content, self.d_pitch, self.d_amplitude, self.d_timbre, self.d_codec, self.n_bins = d_content, d_pitch, d_amplitude, d_timbre, d_codec, n_bins
        self.n_pitch, self.min_pitch, self.max_pitch = n_pitch, min_pitch, max_pitch
        self.n_amplitude, self.min_amplitude, self.max_amplitude = n_amplitude, min_amplitude, max_amplitude
        self.d_model, self.n_heads, self.d_ff, self.n_layers, self.dropout = d_model, n_heads, d_ff, n_layers, dropout

        # Embedding layers for pitch, amplitude features
        self.pitch_embedding = LogEmbedding(n_pitch, d_pitch, min_pitch, max_pitch)
        self.amplitude_embedding = LogEmbedding(n_amplitude, d_amplitude, min_amplitude, max_amplitude)

        # Projection layers for input features
        self.cpa_projection = nn.Linear(d_content + d_pitch + d_amplitude, d_model, bias=True)
        self.d_codec_projection = nn.Linear(d_codec, d_model, bias=True)

        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, d_timbre, dropout) for _ in range(n_layers)
        ])

        # Projection layer for output features
        self.norm = nn.RMSNorm(d_model)
        self.output_projection = nn.Linear(d_model, d_codec * n_bins)

        self.init_params()

    def init_params(self, std: float = 0.02):
        """
        Initialize the parameters of the model using various initialization schemes.
        """
        # Initialize the parameters of embedding layers
        self.pitch_embedding.init_params()
        self.amplitude_embedding.init_params()

        # Initialize the parameters of projection layers
        for linear in [self.cpa_projection, self.d_codec_projection]:
            nn.init.trunc_normal_(linear.weight, std=std)
            if linear.bias is not None:
                nn.init.zeros_(linear.bias)

        nn.init.trunc_normal_(self.output_projection.weight, std=std)
        if self.output_projection.bias is not None:
            nn.init.zeros_(self.output_projection.bias)

        # Initialize the parameters of Transformer blocks
        for i, block in enumerate(self.transformer_blocks):
            block.init_params(std=std/((i + 1) ** 0.5)) # Scale initialization by sqrt of number of layers for stability

    def forward(
        self,
        content: Tensor, pitch: Tensor, amplitude: Tensor, timbre: Tensor,
        content_length: Tensor, pitch_length: Tensor, amplitude_length: Tensor, timbre_length: Tensor,
        target_length: Tensor
    ) -> Tensor:
        """
        Forward pass for the MeanFlowsGenerator.

        Args:
            content (Tensor): Content features of shape (N, T_content, D_content).
            pitch (Tensor): Pitch features of shape (N, T_pitch).
            amplitude (Tensor): Amplitude features of shape (N, T_amplitude).
            timbre (Tensor): Timbre features of shape (N, T_timbre, D_timbre).

            content_length (Tensor): Lengths of content sequences for masking, shape (N,).
            pitch_length (Tensor): Lengths of pitch sequences for masking, shape (N,).
            amplitude_length (Tensor): Lengths of amplitude sequences for masking, shape (N,).
            timbre_length (Tensor): Lengths of timbre sequences for masking, shape (N,).

            target_length (Tensor): Lengths of target sequences for interpolation, shape (N,).

        Returns:
            Tensor: Output tensor of shape (N, N_bins, T, D_codec) for CrossEntropyLoss.
        """
        # assert content.size(0) == pitch.size(0) == amplitude.size(0) == timbre.size(0) == zt.size(0), "Batch size of all input features must be the same."
        # assert (drop_cond is None) or (content.size(0) == drop_cond.size(0)), "Batch size of drop_cond must be the same as other features if provided."
        # assert (content_length is None) or (content.size(0) == content_length.size(0)), "Batch size of content_length must be the same as content if provided."
        # assert (pitch_length is None) or (pitch.size(0) == pitch_length.size(0)), "Batch size of pitch_length must be the same as pitch if provided."
        # assert (amplitude_length is None) or (amplitude.size(0) == amplitude_length.size(0)), "Batch size of amplitude_length must be the same as amplitude if provided."
        # assert (timbre_length is None) or (timbre.size(0) == timbre_length.size(0)), "Batch size of timbre_length must be the same as timbre if provided."
        # assert (zt_length is None) or (zt.size(0) == zt_length.size(0)), "Batch size of zt_length must be the same as zt if provided."

        # assert (content_length is None) or (content_length.max().item() <= content.size(1)), "Max content length cannot exceed content feature length."
        # assert (pitch_length is None) or (pitch_length.max().item() <= pitch.size(1)), "Max pitch length cannot exceed pitch feature length."
        # assert (amplitude_length is None) or (amplitude_length.max().item() <= amplitude.size(1)), "Max amplitude length cannot exceed amplitude feature length."
        # assert (timbre_length is None) or (timbre_length.max().item() <= timbre.size(1)), "Max timbre length cannot exceed timbre feature length."
        # assert (zt_length is None) or (zt_length.max().item() <= zt.size(1)), "Max zt length cannot exceed zt feature length."

        N = target_length.size(0)
        T = target_length.max()

        # Step 1: Interpolate content, pitch and amplitude features to the same temporal resolution as zt (if provided) or the maximum length among them.
        # if content_length is None: content_length = torch.full((N,), content.size(1), dtype=torch.long, device=content.device) # (N,)
        # if pitch_length is None: pitch_length = torch.full((N,), pitch.size(1), dtype=torch.long, device=pitch.device) # (N,)
        # if amplitude_length is None: amplitude_length = torch.full((N,), amplitude.size(1), dtype=torch.long, device=amplitude.device) # (N,)
        # if timbre_length is None: timbre_length = torch.full((N,), timbre.size(1), dtype=torch.long, device=timbre.device) # (N,)
        # if zt_length is None: zt_length: Tensor = torch.maximum(torch.maximum(content_length, pitch_length), amplitude_length).clamp(max=T) # (N,)

        # Create a common destination index grid for the entire batch
        grid: Tensor = torch.arange(T, device=content.device).unsqueeze(0).expand(N, -1) # (N, T)
        # pad_mask: Tensor = grid >= zt_length.unsqueeze(1) # (N, T)

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

        content_interp: Tensor = batched_nearest_interpolate(content, content_length, target_length) # (N, T, D_content)
        pitch_interp: Tensor = batched_nearest_interpolate(pitch.unsqueeze(-1), pitch_length, target_length).squeeze(-1) # (N, T, 1) -> (N, T)
        amplitude_interp: Tensor = batched_nearest_interpolate(amplitude.unsqueeze(-1), amplitude_length, target_length).squeeze(-1) # (N, T, 1) -> (N, T)

        # Step 2: Embed pitch and amplitude features using logarithmic embedding.
        pitch_emb: Tensor = self.pitch_embedding(pitch_interp) # (N, T) -> (N, T, D_pitch)
        amplitude_emb: Tensor = self.amplitude_embedding(amplitude_interp) # (N, T) -> (N, T, D_amplitude)

        # Step 3: Concatenate content, pitch and amplitude features, and project to d_model dimension. Also project timbre features to d_model dimension.
        cpa: Tensor = torch.cat([content_interp, pitch_emb, amplitude_emb], dim=-1) # (N, T, D_content + D_pitch + D_amplitude)
        target: Tensor = self.cpa_projection(cpa) # (N, T, D_model)
        source: Tensor = timbre # (N, T, D_timbre)

        # Step 4: Pass through Transformer blocks with cross-attention to timbre features.
        for block in self.transformer_blocks:
            target: Tensor = block(
                target=target,
                source=source,
                target_length=target_length,
                source_length=timbre_length
            )

        # Step 5: Project the output features back to D_codec dimension.
        target: Tensor = self.norm(target) # (N, T, D_model)
        output: Tensor = self.output_projection(target) # (N, T, D_model) -> (N, T, D_codec * N_bins)
        output: Tensor = output.view(N, T, self.d_codec, self.n_bins).permute(0, 3, 1, 2).contiguous() # (N, T, D_codec * N_bins) -> (N, T, D_codec, N_bins) -> (N, N_bins, T, D_codec)

        return output # (N, N_bins, T, D_codec)