# Embedding class for pitch and amplitude values, based on logarithmic scale.
import math, torch

import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor
from typing import Self


# Function to calculate ALiBi slopes for each attention head based on the original ALiBi paper.
def get_alibi_slopes(n_heads: int) -> list[float]:
    """
    Tính toán hệ số dốc (slopes) m cho từng head theo bài báo ALiBi nguyên bản.
    Đã xử lý chính xác cho cả trường hợp n_heads không phải là lũy thừa của 2.
    """
    def get_slopes_power_of_2(n: int) -> list[float]:
        # Tương đương với 2^(-8/n)
        start = (2 ** (-2 ** -(math.log2(n) - 3))) 
        ratio = start
        return [start * (ratio ** i) for i in range(n)]

    if math.log2(n_heads).is_integer():
        return get_slopes_power_of_2(n_heads)
    else:
        # Ví dụ n_heads = 12
        closest_power_of_2 = 2 ** math.floor(math.log2(n_heads)) # 8
        
        # Tập slopes cơ bản từ 8 heads
        base_slopes = get_slopes_power_of_2(closest_power_of_2) 
        
        # Lấy các slopes xen kẽ mới từ 16 heads bằng [0::2] để tránh trùng lặp
        extra_slopes = get_slopes_power_of_2(2 * closest_power_of_2)[0::2]
        
        # Ghép lại và cắt đúng số lượng head còn thiếu
        return base_slopes + extra_slopes[:n_heads - closest_power_of_2]

# Function to generate cross-attention ALiBi bias for given number of heads and sequence lengths.
def get_cross_alibi_bias(n_heads: int, T: int, S: int, device: torch.device = None) -> Tensor:
    """
    Sinh ra tensor ALiBi (n_heads, T, S) cho Cross-Attention.
    Khoảng cách được tính theo công thức: D(i, j) = |i - j - (T - S)|
    
    Args:
        n_heads (int): Số lượng attention heads.
        T (int): Chiều dài chuỗi Query (Target).
        S (int): Chiều dài chuỗi Key/Value (Source).
        device (torch.device, optional): Device của tensor.

    Returns:
        Tensor: Ma trận ALiBi có kích thước (n_heads, T, S) mang giá trị âm hoặc 0.
    """
    # 1. Tạo lưới tọa độ
    t_idx = torch.arange(T, device=device).unsqueeze(1)  # (T, 1)
    s_idx = torch.arange(S, device=device).unsqueeze(0)  # (1, S)
    
    # 2. Tính ma trận khoảng cách tuyệt đối có dịch chuyển
    distance = torch.abs(t_idx - s_idx - (T - S))  # (T, S)
    
    # 3. Lấy hệ số dốc cho từng head và định dạng lại tensor
    slopes = torch.tensor(get_alibi_slopes(n_heads), device=device).view(n_heads, 1, 1)
    
    # 4. Tính toán ALiBi bias (Luôn mang giá trị <= 0 để phạt trước khi qua Softmax)
    alibi_bias = -slopes * distance  # (n_heads, T, S)
    
    return alibi_bias


# Logarithmic embedding for pitch and amplitude values.
class LogEmbedding(nn.Module):
    """
    Embedding class for pitch and amplitude values, based on logarithmic scale.
    """
    def __init__(self, n_bins: int, d_embedding: int, min_value: float, max_value: float):
        super().__init__()
        self.n_bins = n_bins
        self.d_embedding = d_embedding
        self.min_value = min_value
        self.max_value = max_value
        self.embedding = nn.Embedding(n_bins, d_embedding)
        
        # Precompute log(min_value) and log(max_value) for scaling
        self.register_buffer('log_min', torch.tensor(math.log(min_value)))
        self.register_buffer('log_max', torch.tensor(math.log(max_value)))

    def init_params(self, mean: float = 0.0, std: float = 0.02):
        """
        Initialize embedding parameters using truncated normal initialization.
        """
        nn.init.trunc_normal_(self.embedding.weight, mean=mean, std=std)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass to compute the log-scaled embedding.
        
        Args:
            x (Tensor): Input tensor of shape (B, T) containing pitch or amplitude values.
        
        Returns:
            Tensor: Log-scaled embeddings of shape (B, T, embedding_dim).
        """
        # Tách riêng mask của các khung vô thanh (x < self.min_value) để gán bin 0 sau này
        unvoiced_mask = (x < self.min_value)
        
        # Clamp các giá trị có thanh
        x_clamped = torch.clamp(x, min=self.min_value, max=self.max_value)
        
        # Phép tính trên tensor x (giữ nguyên device hiện tại của x)
        log_x = torch.log(x_clamped)
        
        # Ánh xạ log_x vào khoảng từ bin 1 đến n_bins - 1
        scaled_log_x = (log_x - self.log_min) / (self.log_max - self.log_min) * (self.n_bins - 2)
        
        # Làm tròn và tịnh tiến lên 1 đơn vị (chừa bin 0 cho vô thanh)
        bin_indices = torch.round(scaled_log_x).long() + 1
        bin_indices = torch.clamp(bin_indices, min=1, max=self.n_bins - 1)
        
        # Gán bin 0 cho các vị trí vô thanh
        bin_indices = torch.where(unvoiced_mask, torch.zeros_like(bin_indices), bin_indices)
        
        # Lấy embedding từ bin indices
        embeddings = self.embedding(bin_indices)
        return embeddings


# SwiGLU activation function module, used in the feedforward layers of the DiT block.
class SwiGLU(nn.Module):
    """
    SwiGLU activation function module, used in the feedforward layers of the DiT block.
    """
    def __init__(self, d_in: int, d_hidden: int):
        """
        Initialize the SwiGLU module with two linear layers for gating and up-projection.

        Args:
            d_in (int): Input dimensionality.
            d_hidden (int): Hidden dimensionality for the SwiGLU activation.
        """
        super().__init__()
        # 2 ma trận tham số (W và V). Bias thường được set = False trong các LLMs.
        self.w_gate = nn.Linear(d_in, d_hidden, bias=False)
        self.w_up = nn.Linear(d_in, d_hidden, bias=False)

        self.init_params()

    def init_params(self, std: float = 0.02):
        """
        Initialize the parameters of the SwiGLU module using truncated normal initialization.
        """
        for linear in [self.w_gate, self.w_up]:
            nn.init.trunc_normal_(linear.weight, std=std)
            if linear.bias is not None:
                nn.init.zeros_(linear.bias)

    def forward(self, x: Tensor) -> Tensor:
        """
        Perform the SwiGLU activation function.

        Args:
            x (Tensor): Input tensor of shape (batch_size, seq_len, d_in)
        
        Returns:
            Tensor: Output tensor of shape (batch_size, seq_len, d_hidden) after applying the SwiGLU activation.
        """
        return F.silu(self.w_gate(x)) * self.w_up(x)
    

# MLP module that combines the SwiGLU activation with a final linear projection, used in the feedforward layers of the DiT block.
class MLP(nn.Module):
    """
    MLP module that combines the SwiGLU activation with a final linear projection, used in the feedforward layers of the DiT block.
    """
    def __init__(self, d_in: int, d_hidden: int, d_out: int, dropout: float = 0.1):
        """
        Initialize the MLP module with a SwiGLU layer followed by a linear projection.
        
        Args:
            d_in (int): Input dimensionality.
            d_hidden (int): Hidden dimensionality for the SwiGLU activation.
            d_out (int): Output dimensionality after the final linear projection.
            dropout (float): Dropout probability for regularization. Default is 0.1.
        """
        super().__init__()
        # Khởi tạo cụm SwiGLU (chứa 2 ma trận W_gate và W_up)
        self.swiglu = SwiGLU(d_in, d_hidden)

        # Thêm lớp dropout sau SwiGLU để tăng cường regularization
        self.dropout = nn.Dropout(dropout)

        # Ma trận thứ 3 để chiếu kết quả về lại kích thước out_features
        self.w_down = nn.Linear(d_hidden, d_out, bias=False)

    def init_params(self, std: float = 0.02):
        """
        Initialize the parameters of the MLP module using truncated normal initialization.
        """
        self.swiglu.init_params(std=std)
        nn.init.trunc_normal_(self.w_down.weight, std=std)
        if self.w_down.bias is not None:
            nn.init.zeros_(self.w_down.bias)

    def forward(self, x: Tensor) -> Tensor:
        """
        Perform the forward pass through the MLP module, applying the SwiGLU activation followed by a linear projection.
        
        Args:
            x (Tensor): Input tensor of shape (batch_size, seq_len, d_in)
        
        Returns:
            Tensor: Output tensor of shape (batch_size, seq_len, d_out) after applying the MLP transformation.
        """
        # Bước 1: Đi qua SwiGLU
        hidden_states = self.swiglu(x)
        # Bước 2: Đi qua lớp dropout
        hidden_states = self.dropout(hidden_states)
        # Bước 3: Đi qua lớp chiếu cuối cùng
        out = self.w_down(hidden_states)
        return out
    

class MultiheadSelfAttentionWithALiBi(nn.Module):
    """
    Multi-head self-attention module that incorporates ALiBi (Attention with Linear Biases) for enhanced positional encoding.
    """
    def __init__(
        self,
        d_model: int, n_heads: int, dropout: float = 0.1,
        d_query: int | None = None, d_key: int | None = None, d_value: int | None = None
    ):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads."

        # Parameters for the multi-head self-attention
        self.d_model = d_model
        self.n_heads = n_heads
        self.dropout = dropout
        self._dropout = dropout

        self.d_query = d_query or d_model
        self.d_key = d_key or d_model
        self.d_value = d_value or d_model

        # Multi-head attention layer
        self.q_project = nn.Linear(self.d_query, d_model, bias=False)
        self.k_project = nn.Linear(self.d_key, d_model, bias=False)
        self.v_project = nn.Linear(self.d_value, d_model, bias=False)
        self.out_project = nn.Linear(d_model, self.d_query, bias=False)

        self.init_params()

    def init_params(self, std: float = 0.02):
        """
        Initialize the parameters of the MultiheadSelfAttentionWithALiBi module using truncated normal initialization.
        """
        nn.init.trunc_normal_(self.q_project.weight, std=std)
        nn.init.trunc_normal_(self.k_project.weight, std=std)
        nn.init.trunc_normal_(self.v_project.weight, std=std)
        nn.init.trunc_normal_(self.out_project.weight, std=std)

    def train(self, mode: bool = True) -> Self:
        """
        Override the default train() method to ensure that the ALiBi module is also set to training mode when the parent module is set to training mode.
        
        Args:
            mode (bool): If True, sets the module in training mode. If False, sets it in evaluation mode.
        
        Returns:
            Self: The module itself after setting the training mode.
        """
        super().train(mode)
        self._dropout = self.dropout if mode else 0.0
        return self

    def forward(self, x: Tensor, key_padding_mask: Tensor, is_causal: bool = False) -> Tensor:
        """
        Forward pass for the multi-head self-attention with RoPE.

        Args:
            x (Tensor): Input tensor of shape (N, T, D).
            key_padding_mask (Tensor): Boolean mask for padded positions, shape (N, T).
            is_causal (bool): Whether to apply causal masking. Default is False.

        Returns:
            Tensor: Output tensor of shape (N, T, D_model) after applying multi-head attention with RoPE.
        """
        N, T, _ = x.shape
        device = x.device
        d_head = self.d_model // self.n_heads

        # Step 1: Project to query, key, value spaces
        q = self.q_project(query)  # (N, T, D_model)
        k = self.k_project(key)  # (N, S, D_model)
        v = self.v_project(value)  # (N, S, D_model)

        # Step 2: Reshape and transpose for multi-head attention
        q = q.view(N, T, self.n_heads, d_head).transpose(1, 2)  # (N, N_heads, T, D_head)
        k = k.view(N, T, self.n_heads, d_head).transpose(1, 2)  # (N, N_heads, T, D_head)
        v = v.view(N, T, self.n_heads, d_head).transpose(1, 2)  # (N, N_heads, T, D_head)

        # Step 3: Generate and apply ALiBi biases
        alibi_bias = get_cross_alibi_bias(self.n_heads, T, S, device=device)  # (n_heads, T, S)

        # Step 4: Prepare attention mask as an additive float mask
        attn_mask = torch.zeros((N, 1, 1, T), dtype=q.dtype, device=device)
        attn_mask = attn_mask.masked_fill(key_padding_mask.view(N, 1, 1, T), float("-inf"))
        if is_causal:
            causal_mask = torch.triu(torch.full((T, T), float("-inf"), dtype=q.dtype, device=device), diagonal=1)  # (T, T)
            attn_mask = attn_mask + causal_mask # (N, 1, T, T)

        # Step 5: Compute scaled dot-product attention
        attn_output = F.scaled_dot_product_attention(
            query=q,
            key=k,
            value=v,
            attn_mask=attn_mask,
            dropout_p=self._dropout
        ) # (N, N_heads, T, D_head)

        # Step 6: Reproject output
        attn_output = attn_output.transpose(1, 2).contiguous().view(N, T, self.d_model) # (N, T, D_model)
        output = self.out_project(attn_output) # (N, T, D_query)

        return output


class TransformerBlock(nn.Module):
    """
    Transformer Encoder-like block module that combines Transformer Encoder layers with RoPE.
    """
    def __init__(
        self,
        d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1
    ):
        """
        Initialize the TransformerBlock with specified parameters for the Transformer Encoder and conditioning dimensions.

        Args:
            d_model (int): Dimensionality of the model (input/output of the Transformer layers).
            n_heads (int): Number of attention heads in the multi-head attention mechanism.
            d_ff (int): Dimensionality of the feed-forward network.
            dropout (float): Dropout probability.
        """
        super().__init__()
        # Constants for the Transformer Block
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_ff = d_ff
        self.dropout = dropout

        # AdaRMSN-Zero conditioning projection for timbre embedding
        self.timbre_mlp = MLP(d_timbre, d_model, 6 * d_model, dropout)

        # Multi-Head Self-Attention Layer
        self.norm_1 = nn.RMSNorm(d_model, elementwise_affine=False)
        self.self_attention = MultiheadSelfAttentionWithALiBi(d_model, n_heads, dropout=dropout)

        # Feed-Forward Network (MLP)
        self.norm_2 = nn.RMSNorm(d_model)
        self.mlp = MLP(d_model, d_ff, d_model, dropout)

        # Initialize parameters
        self.init_params()

    def init_params(self, std: float = 0.02):
        """
        Initialize the parameters of the TransformerBlock using zero initialization for linear layers and truncated normal initialization for attention layers.
        """
        # Initialize attention layers using truncated normal initialization
        self.self_attention.init_params(std=std)

        # Initialize MLP layers
        self.mlp.init_params(std=std)

        # Initialize timbre MLP layers
        self.timbre_mlp.init_params(std=std)
        nn.init.zeros_(self.timbre_mlp.w_down.weight)

    def forward(
        self,
        input: Tensor, input_length: Tensor
    ) -> Tensor:
        """
        Forward pass for the Transformer Block.

        Args:
            input (Tensor): Input tensor of shape (N, T, D_model), came from the (content + pitch + amplitude + token embeddings) conditioning.
            input_length (Tensor): Lengths of the input sequences (N,).

        Returns:
            Tensor: Output tensor of shape (N, T, D_model) after processing through the Transformer block.
        """
        N, T, _ = input.shape

        # Step 1: Create padding masks for input sequences (True for padding positions, False for valid positions)
        pad_mask = torch.arange(T, device=input.device).unsqueeze(0) >= input_length.unsqueeze(1) # (N, T)

        # Step 2: Multi-Head Self-Attention
        normed_target_1: Tensor = self.norm_1(input) # (N, T, D_model)
        attn_output: Tensor = self.self_attention(
            x=normed_target_1,
            key_padding_mask=pad_mask,
            is_causal=False
        ) # (N, T, D_model)
        input = input + attn_output # (N, T, D_model)

        # Step 3: Feed-Forward Network (MLP)
        normed_target_2: Tensor = self.norm_2(input) # (N, T, D_model)
        mlp_output: Tensor = self.mlp(normed_target_2) # (N, T, D_model)
        input = input + mlp_output # (N, T, D_model)

        return input # (N, T, D_model)