# VoiceGenerator network
import torch
import torch.nn as nn

from torch import Tensor
from .submodules import LogEmbedding, TransformerBlock


class VoiceGenerator(nn.Module):
    """
    VoiceGenerator is a neural network module designed for voice conversion tasks.
    It consists of multiple blocks that process input features and generate converted features.
    """
    def __init__(
        self,
        d_content: int, d_pitch: int, d_amplitude: int, d_token: int, d_fsq: int, n_bins: int,
        n_pitch: int, min_pitch: float, max_pitch: float, n_amplitude: int, min_amplitude: float, max_amplitude: float,
        d_model: int, n_heads: int, d_ff: int, n_layers: int, dropout: float = 0.1,
        fsq_weight: str | Tensor | None = None, token_weight: str | Tensor | None = None
    ):
        """
        Initialize the VoiceGenerator module.

        Args:
            d_content (int): The dimensionality of the content embedding (came from VietASR content features). Should be 512.
            d_pitch (int): The dimensionality of the pitch embedding (after logarithmic embedding).
            d_amplitude (int): The dimensionality of the amplitude embedding (after logarithmic embedding).
            d_token (int): The dimensionality of the token embedding (came from NeuCodec token features). Should be 1024.
            d_fsq (int): The dimensionality of the finite scalar quantization embedding. Should be 8.
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

            fsq_weight (str | Tensor | None): Optional pre-trained weights for the FSQ embedding layer. Should have shape (n_bins ** d_fsq, d_fsq) if provided.
            token_weight (str | Tensor | None): Optional pre-trained weights for the FSQ projection layer. Should have shape (d_fsq, d_token) if provided.
        """
        super().__init__()
        # Initialize model parameters
        self.d_content, self.d_pitch, self.d_amplitude = d_content, d_pitch, d_amplitude
        self.d_token, self.d_fsq, self.n_bins = d_token, d_fsq, n_bins

        self.n_pitch, self.min_pitch, self.max_pitch = n_pitch, min_pitch, max_pitch
        self.n_amplitude, self.min_amplitude, self.max_amplitude = n_amplitude, min_amplitude, max_amplitude

        self.d_model, self.n_heads, self.d_ff, self.n_layers, self.dropout = d_model, n_heads, d_ff, n_layers, dropout

        # Embedding layer for tokens from NeuCodec
        self.fsq_embedding = nn.Embedding(n_bins ** d_fsq, d_fsq)
        self.fsq_projection = nn.Linear(d_fsq, d_token)
        self.token_projection = nn.Linear(d_token, d_model) if d_token != d_model else nn.Identity()
        self.mask_token = nn.Parameter(torch.zeros(d_token))

        # Embedding layers for pitch, amplitude features
        self.pitch_embedding = LogEmbedding(n_pitch, d_pitch, min_pitch, max_pitch)
        self.amplitude_embedding = LogEmbedding(n_amplitude, d_amplitude, min_amplitude, max_amplitude)

        # Projection layers for input features
        self.cpa_projection = nn.Linear(d_content + d_pitch + d_amplitude, d_model)

        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)
        ])

        # Projection layer for output features
        self.norm = nn.RMSNorm(d_model)
        self.output_projection = nn.Linear(d_model, n_bins * d_fsq)

        self.init_params(fsq_weight=fsq_weight, token_weight=token_weight)

    def init_params(self, std: float = 0.02, fsq_weight: str | Tensor | None = None, token_weight: str | Tensor | None = None):
        """
        Initialize the parameters of the model using various initialization schemes.
        """
        # Initialize the parameters of embedding layers
        self.pitch_embedding.init_params()
        self.amplitude_embedding.init_params()
        nn.init.trunc_normal_(self.mask_token, std=std)

        # Copy pre-trained weights for embedding layers if provided
        if fsq_weight is not None:
            if isinstance(fsq_weight, str):
                fsq_weight = torch.load(fsq_weight)
            with torch.no_grad():
                self.fsq_embedding.weight.copy_(fsq_weight)
                self.fsq_embedding.weight.requires_grad = False # Freeze the fsq embedding weights
        else:
            nn.init.trunc_normal_(self.fsq_embedding.weight, std=std)

        if token_weight is not None:
            if isinstance(token_weight, str):
                token_weight = torch.load(token_weight)
            with torch.no_grad():
                self.fsq_projection.weight.copy_(token_weight)
                self.fsq_projection.weight.requires_grad = False # Freeze the fsq projection weights
        else:
            nn.init.trunc_normal_(self.fsq_projection.weight, std=std)
            if self.fsq_projection.bias is not None:
                nn.init.zeros_(self.fsq_projection.bias)

        # Initialize the parameters of projection layers
        for linear in [self.cpa_projection, self.token_projection, self.output_projection]:
            if isinstance(linear, nn.Linear):
                nn.init.trunc_normal_(linear.weight, std=std)
                if linear.bias is not None:
                    nn.init.zeros_(linear.bias)

        # Initialize the parameters of Transformer blocks
        for i, block in enumerate(self.transformer_blocks):
            block.init_params(std=std/((i + 1) ** 0.5)) # Scale initialization by sqrt of number of layers for stability

    def forward(
        self,
        content: Tensor, pitch: Tensor, amplitude: Tensor, token: Tensor,
        mask_indices: Tensor, content_length: Tensor, token_length: Tensor
    ) -> Tensor:
        """
        Forward pass for the VoiceGenerator.

        Args:
            content (Tensor): Content features of shape (N, T', D_content). T' ~ T // 2.
            pitch (Tensor): Pitch features of shape (N, T).
            amplitude (Tensor): Amplitude features of shape (N, T).
            token (Tensor): Token tensor for teacher forcing of shape (N, T).

            mask_indices (Tensor): Indices of tokens to be masked, shape (N, T). True for positions to be masked, False for positions to be kept.
            content_length (Tensor): Lengths of content sequences for masking, shape (N,).
            token_length (Tensor): Lengths of token sequences for masking, which is also pitch & amplitude lengths, shape (N,).

        Returns:
            Tensor: Output tensor of shape (N, N_bins, T, D_fsq) for CrossEntropyLoss.
        """
        N, T = token.shape

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
        content_interp: Tensor = batched_nearest_interpolate(content, content_length, token_length) # (N, T, D_content)

        # Step 2: Embed pitch and amplitude features using logarithmic embedding. Embed target tokens using token embedding and project to d_model dimension if necessary.
        pitch_emb: Tensor = self.pitch_embedding(pitch) # (N, T) -> (N, T, D_pitch)
        amplitude_emb: Tensor = self.amplitude_embedding(amplitude) # (N, T) -> (N, T, D_amplitude)

        fsq_emb: Tensor = self.fsq_embedding(token) # (N, T) -> (N, T, D_fsq)
        token_emb: Tensor = self.fsq_projection(fsq_emb) # (N, T, D_fsq) -> (N, T, D_token)

        # Apply mask token based on mask_indices
        token_emb: Tensor = torch.where(mask_indices.unsqueeze(-1).bool(), self.mask_token, token_emb) # (N, T, D_token)
        token_emb: Tensor = self.token_projection(token_emb) # (N, T, D_token) -> (N, T, D_model)

        # Step 3: Concatenate content, pitch and amplitude features, and project to d_model dimension.
        cpa: Tensor = torch.cat([content_interp, pitch_emb, amplitude_emb], dim=-1) # (N, T, D_content + D_pitch + D_amplitude)
        cpa_emb: Tensor = self.cpa_projection(cpa) # (N, T, D_content + D_pitch + D_amplitude) -> (N, T, D_model)
        input: Tensor = token_emb + cpa_emb # (N, T, D_model)

        # Step 4: Pass through Transformer blocks with cross-attention to timbre features.
        for block in self.transformer_blocks:
            input: Tensor = block(
                input=input,
                input_length=token_length
            ) # (N, T, D_model)

        # Step 5: Project the output features back to D_codec dimension.
        output: Tensor = self.norm(input) # (N, T, D_model)
        output: Tensor = self.output_projection(output) # (N, T, D_model) -> (N, T, N_bins * D_fsq)
        output = output.view(N, T, self.n_bins, self.d_fsq) # (N, T, N_bins * D_fsq) -> (N, T, N_bins, D_fsq)
        output = output.permute(0, 2, 1, 3) # (N, T, N_bins, D_fsq) -> (N, N_bins, T, D_fsq) for CrossEntropyLoss with token shape (N, T, D_fsq)

        return output # (N, N_bins, T, D_fsq)