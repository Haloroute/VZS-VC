# MeanFlows network
import torch

import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor
from .submodules import DiTBlock, LogEmbedding, MLP, SinusoidalPositionalEncoding


class MeanFlowsGenerator(nn.Module):
    """
    MeanFlowsGenerator is a neural network module designed for voice conversion tasks.
    It consists of multiple MeanFlows blocks that process input features and generate converted features based on the given time step.
    The architecture is inspired by the MeanFlows model, which incorporates attention mechanisms and MLP layers with AdaRMSN-Zero conditioning.
    """
    def __init__(
        self,
        d_time: int, d_timbre: int, d_content: int, d_pitch: int, d_amplitude: int, d_codec: int,
        n_pitch: int, min_pitch: float, max_pitch: float, n_amplitude: int, min_amplitude: float, max_amplitude: float,
        d_model: int, n_heads: int, d_ff: int, n_layers: int, dropout: float = 0.1
    ):
        """
        Initialize the MeanFlowsGenerator module.
        
        Args:
            d_time (int): The dimensionality of the time embedding (after sinusoidal encoding + MLP). 
            d_timbre (int): The dimensionality of the timbre embedding (came from NeuCodec acoustic features). Should be 1024.
            d_content (int): The dimensionality of the content embedding (came from VietASR content features). Should be 512.
            d_pitch (int): The dimensionality of the pitch embedding (after logarithmic embedding).
            d_amplitude (int): The dimensionality of the amplitude embedding (after logarithmic embedding).
            d_codec (int): The dimensionality of the codec embedding (used for Finite Scalar Quantization).

            n_pitch (int): The number of bins for pitch embedding.
            min_pitch (float): The minimum value for pitch embedding (should be a positive value). Should be around 32.7 (C1 note).
            max_pitch (float): The maximum value for pitch embedding (should be a positive value). Should be around 1244.5 (D#6 note).
            n_amplitude (int): The number of bins for amplitude embedding.
            min_amplitude (float): The minimum value for amplitude embedding (should be a positive value). Should be around 0.01.
            max_amplitude (float): The maximum value for amplitude embedding (should be a positive value). Should be around 1.0.
            time_scale (float): The scaling factor for time before passing through sinusoidal encoding. Default is 1000.0.

            d_model (int): The dimensionality of the model (feature dimension).
            n_heads (int): The number of attention heads in each DiT block.
            d_ff (int): The dimensionality of the feed-forward layer in each DiT block.
            n_layers (int): The number of DiT blocks in the generator.
            dropout (float): The dropout rate for regularization. Default is 0.1.
        """
        super().__init__()
        # Initialize model parameters
        self.d_time, self.d_timbre, self.d_content, self.d_pitch, self.d_amplitude, self.d_codec = d_time, d_timbre, d_content, d_pitch, d_amplitude, d_codec
        self.n_pitch, self.min_pitch, self.max_pitch = n_pitch, min_pitch, max_pitch
        self.n_amplitude, self.min_amplitude, self.max_amplitude = n_amplitude, min_amplitude, max_amplitude
        self.d_model, self.n_heads, self.d_ff, self.n_layers, self.dropout = d_model, n_heads, d_ff, n_layers, dropout

        # Embedding layers for pitch, amplitude and time features
        self.pitch_embedding = LogEmbedding(n_pitch, d_pitch, min_pitch, max_pitch)
        self.amplitude_embedding = LogEmbedding(n_amplitude, d_amplitude, min_amplitude, max_amplitude)
        self.time_embedding = SinusoidalPositionalEncoding(d_time)

        # Projection layers for input features
        self.time_projection = MLP(2 * d_time, 2 * d_time, d_time, dropout)
        self.cpa_projection = nn.Linear(d_content + d_pitch + d_amplitude, d_model, bias=True)
        self.d_codec_projection = nn.Linear(d_codec, d_model, bias=True)

        # DiT blocks
        self.dit_blocks = nn.ModuleList([
            DiTBlock(d_model, n_heads, d_ff, d_time, d_timbre, dropout) for _ in range(n_layers)
        ])

        # Projection layer for output features
        self.norm = nn.RMSNorm(d_model)
        self.output_projection = nn.Linear(d_model, d_codec)

        self.init_params()

    def init_params(self, std: float = 0.02):
        """
        Initialize the parameters of the model using various initialization schemes.
        """
        # Initialize the parameters of embedding layers
        self.pitch_embedding.init_params()
        self.amplitude_embedding.init_params()

        # Initialize the parameters of projection layers
        self.time_projection.init_params()
        for linear in [self.cpa_projection, self.d_codec_projection]:
            nn.init.trunc_normal_(linear.weight, std=std)
            if linear.bias is not None:
                nn.init.zeros_(linear.bias)

        nn.init.zeros_(self.output_projection.weight)
        if self.output_projection.bias is not None:
            nn.init.zeros_(self.output_projection.bias)

        # Initialize the parameters of DiT blocks
        for i, block in enumerate(self.dit_blocks):
            block.init_params(std=std/((i + 1) ** 0.5)) # Scale initialization by sqrt of number of layers for stability

    def forward(
        self,
        content: Tensor, pitch: Tensor, amplitude: Tensor, timbre: Tensor, start_timestep: Tensor, end_timestep: Tensor, pre_vq: Tensor | None = None,
        content_length: Tensor | None = None, pitch_length: Tensor | None = None, amplitude_length: Tensor | None = None, timbre_length: Tensor | None = None,
        pre_vq_length: Tensor | None = None, drop_conditioning: Tensor | None = None
    ) -> Tensor:
        """
        Forward pass for the MeanFlowsGenerator.
        Args:
            content (Tensor): Content features of shape (N, T_content, D_content).
            pitch (Tensor): Pitch features of shape (N, T_pitch).
            amplitude (Tensor): Amplitude features of shape (N, T_amplitude).
            timbre (Tensor): Timbre features of shape (N, T_timbre, D_timbre).

            start_timestep (Tensor): Start time step (r) tensor of shape (N,). Note that 0.0 <= r <= t <= 1.0.
            end_timestep (Tensor): End time step (t) tensor of shape (N,). Note that 0.0 <= r <= t <= 1.0.
            pre_vq (Tensor | None): Pre-VQ features for teacher forcing during training, shape (N, T, D_codec). Should be None during inference.

            content_length (Tensor | None): Lengths of content sequences for masking, shape (N,). Should be None if no masking is required.
            pitch_length (Tensor | None): Lengths of pitch sequences for masking, shape (N,). Should be None if no masking is required.
            amplitude_length (Tensor | None): Lengths of amplitude sequences for masking, shape (N,). Should be None if no masking is required.
            timbre_length (Tensor | None): Lengths of timbre sequences for masking, shape (N,). Should be None if no masking is required.
            pre_vq_length (Tensor | None): Lengths of pre-VQ sequences for masking, shape (N,). Should be None if no masking is required.
        
            drop_conditioning (Tensor | None): Optional tensor of shape (N,) indicating which samples in the batch should have their conditioning features dropped for regularization.
            Should be binary (0 or 1) and can be None if no conditioning dropout is applied.
        Returns:
            Tensor: Output tensor of shape (N, T, D_codec).
        """
        assert content.size(0) == pitch.size(0) == amplitude.size(0) == timbre.size(0), "Batch size of all input features must be the same."
        assert (pre_vq is None) or (content.size(0) == pre_vq.size(0)), "Batch size of pre_vq must be the same as other features if provided."
        assert (drop_conditioning is None) or (content.size(0) == drop_conditioning.size(0)), "Batch size of drop_conditioning must be the same as other features if provided."
        assert (content_length is None) or (content.size(0) == content_length.size(0)), "Batch size of content_length must be the same as content if provided."
        assert (pitch_length is None) or (pitch.size(0) == pitch_length.size(0)), "Batch size of pitch_length must be the same as pitch if provided."
        assert (amplitude_length is None) or (amplitude.size(0) == amplitude_length.size(0)), "Batch size of amplitude_length must be the same as amplitude if provided."
        assert (timbre_length is None) or (timbre.size(0) == timbre_length.size(0)), "Batch size of timbre_length must be the same as timbre if provided."
        assert (pre_vq_length is None) or (pre_vq is not None and pre_vq.size(0) == pre_vq_length.size(0)), "Batch size of pre_vq_length must be the same as pre_vq if provided."

        assert (content_length is None) or (content_length.max().item() <= content.size(1)), "Max content length cannot exceed content feature length."
        assert (pitch_length is None) or (pitch_length.max().item() <= pitch.size(1)), "Max pitch length cannot exceed pitch feature length."
        assert (amplitude_length is None) or (amplitude_length.max().item() <= amplitude.size(1)), "Max amplitude length cannot exceed amplitude feature length."
        assert (timbre_length is None) or (timbre_length.max().item() <= timbre.size(1)), "Max timbre length cannot exceed timbre feature length."
        assert (pre_vq_length is None) or (pre_vq is not None and pre_vq_length.max().item() <= pre_vq.size(1)), "Max pre_vq length cannot exceed pre_vq feature length."
        
        N, _, _ = content.shape

        # Step 1: Interpolate content, pitch and amplitude features to the same temporal resolution as pre_vq (if provided) or the maximum length among them.
        if content_length is None: content_length = torch.full((N,), content.size(1), dtype=torch.long, device=content.device) # (N,)
        if pitch_length is None: pitch_length = torch.full((N,), pitch.size(1), dtype=torch.long, device=pitch.device) # (N,)
        if amplitude_length is None: amplitude_length = torch.full((N,), amplitude.size(1), dtype=torch.long, device=amplitude.device) # (N,)
        if timbre_length is None: timbre_length = torch.full((N,), timbre.size(1), dtype=torch.long, device=timbre.device) # (N,)
        if pre_vq is not None and pre_vq_length is None: pre_vq_length = torch.full((N,), pre_vq.size(1), dtype=torch.long, device=pre_vq.device) # (N,)

        if pre_vq is not None:
            interpolated_length: Tensor = pre_vq_length # (N,)
            max_interp_len: int = pre_vq.size(1)
        else:
            interpolated_length: Tensor = torch.max(torch.max(content_length, pitch_length), amplitude_length) # (N,)
            max_interp_len: int = interpolated_length.max().item()

        # Create a common destination index grid for the entire batch
        grid: Tensor = torch.arange(max_interp_len, device=content.device).unsqueeze(0).expand(N, -1) # (N, max_interp_len)
        pad_mask: Tensor = grid >= interpolated_length.unsqueeze(1) # (N, max_interp_len)

        def batched_nearest_interpolate(x: Tensor, src_len: Tensor, dst_len: Tensor) -> Tensor:
            """
            Perform batched nearest-neighbor interpolation on 2D or 3D sequences using index gathering.

            Args:
                x (Tensor): Input tensor of shape (N, T_src) for 2D or (N, T_src, D) for 3D.
                src_len (Tensor): Source sequence lengths of shape (N,).
                dst_len (Tensor): Target sequence lengths of shape (N,).

            Returns:
                Tensor: Interpolated tensor of shape (N, max_interp_len) for 2D or (N, max_interp_len, D) for 3D.
            """
            # Calculate the nearest mapping ratio: index = floor(grid * (src_len / dst_len))
            ratio: Tensor = src_len.float() / dst_len.float() # (N,)
            src_indices: Tensor = (grid.float() * ratio.unsqueeze(1)).long() # (N, max_interp_len)

            # Limit indices to prevent out-of-bounds errors in the padding region
            max_idx: Tensor = (src_len.unsqueeze(1) - 1).clamp(min=0) # (N, 1)
            src_indices: Tensor = torch.clamp(src_indices, min=torch.zeros_like(max_idx), max=max_idx) # (N, max_interp_len)

            if x.dim() == 3:
                src_indices_expanded: Tensor = src_indices.unsqueeze(-1).expand(-1, -1, x.size(-1)) # (N, max_interp_len, D_content)
                out: Tensor = torch.gather(x, dim=1, index=src_indices_expanded) # (N, max_interp_len, D_content)
                out[pad_mask] = 0.0 # (N, max_interp_len, D_content)
            else:
                out: Tensor = torch.gather(x, dim=1, index=src_indices) # (N, max_interp_len)
                out[pad_mask] = 0.0 # (N, max_interp_len)
            
            return out

        content_interp: Tensor = batched_nearest_interpolate(content, content_length, interpolated_length) # (N, max_interp_len, D_content)
        pitch_interp: Tensor = batched_nearest_interpolate(pitch, pitch_length, interpolated_length) # (N, max_interp_len)
        amplitude_interp: Tensor = batched_nearest_interpolate(amplitude, amplitude_length, interpolated_length) # (N, max_interp_len)

        # Step 2: Embed pitch and amplitude features using logarithmic embedding, and time features using sinusoidal encoding + MLP.
        pitch_emb: Tensor = self.pitch_embedding(pitch_interp) # (N, T) -> (N, T, D_pitch)
        amplitude_emb: Tensor = self.amplitude_embedding(amplitude_interp) # (N, T) -> (N, T, D_amplitude)

        # Scale timestep [0, 1] to range [0, 999] to get sinusoidal encoding
        start_indices: Tensor = (torch.minimum(start_timestep, end_timestep) * 999.0).long().clamp(0, 999) # r: (N,)
        elapse_indices: Tensor = (torch.abs(end_timestep - start_timestep) * 999.0).long().clamp(0, 999) # t - r: (N,)
        start_emb: Tensor = self.time_embedding(start_indices)  # (N, D_time)
        elapse_emb: Tensor = self.time_embedding(elapse_indices)  # (N, D_time)
        time_emb: Tensor = torch.cat([start_emb, elapse_emb], dim=-1)  # (N, 2 * D_time)
        time_emb: Tensor = self.time_projection(time_emb)  # (N, D_time)

        # Step 3: Create noise-imbued pre_vq features by adding noise to pre_vq (if provided) or creating noise from scratch (if pre_vq is None).
        if pre_vq is not None:
            t: Tensor = torch.maximum(end_timestep, start_timestep).view(N, 1, 1)
            noise: Tensor = torch.randn_like(pre_vq) # (N, T, D_codec)
            z_t: Tensor = (t * noise) + ((1 - t) * pre_vq) # (N, T, D_codec)
        else:
            z_t: Tensor = torch.randn(N, max_interp_len, self.d_codec, device=content.device) # (N, T, D_codec)

        z_t[pad_mask] = 0.0 # (N, T, D_codec)

        # Step 4: Concatenate content, pitch and amplitude features, and project to d_model dimension. Also project timbre features to d_model dimension.
        cpa: Tensor = torch.cat([content_interp, pitch_emb, amplitude_emb], dim=-1) # (N, T, D_content + D_pitch + D_amplitude)
        source: Tensor = timbre # (N, T, D_timbre)

        # Step 4.5: Apply conditioning dropout if specified
        if drop_conditioning is not None and drop_conditioning.any():
            # Ép về dạng (N, 1, 1) để tận dụng PyTorch Broadcasting
            drop_mask: Tensor = drop_conditioning.view(-1, 1, 1).bool() # (N, 1, 1)
            
            # Masked_fill sẽ tự động khớp kích thước với cpa và timbre
            cpa = cpa.masked_fill(drop_mask, 0.0) # (N, T, D_content + D_pitch + D_amplitude)
            source = source.masked_fill(drop_mask, 0.0) # (N, T, D_timbre)

        cpa_proj: Tensor = self.cpa_projection(cpa) # (N, T, D_model)
        z_t_proj: Tensor = self.d_codec_projection(z_t) # (N, T, D_model)

        target: Tensor = cpa_proj + z_t_proj # (N, T, D_model)

        # Step 5: Pass through DiT blocks with time conditioning and cross-attention to timbre features.
        for block in self.dit_blocks:
            target: Tensor = block(
                target=target,
                source=source,
                timestep=time_emb,
                target_length=interpolated_length,
                source_length=timbre_length
            )

        # Step 6: Project the output features back to d_codec dimension.
        target: Tensor = self.norm(target) # (N, T, D_model)
        output: Tensor = self.output_projection(target) # (N, T, D_model) -> (N, T, D_codec)

        return output