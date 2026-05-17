# Embedding class for pitch and amplitude values, based on logarithmic scale.
import math, torch
import torch.nn as nn


# Logarithmic embedding for pitch and amplitude values.
class LogEmbedding(nn.Module):
    """
    Embedding class for pitch and amplitude values, based on logarithmic scale.
    """
    def __init__(self, n_bins: int, d_embedding: int, min_value: float, max_value: float):
        super(LogEmbedding, self).__init__()
        self.n_bins = n_bins
        self.d_embedding = d_embedding
        self.min_value = min_value
        self.max_value = max_value
        self.embedding = nn.Embedding(n_bins, d_embedding)
        
        # Precompute log(min_value) and log(max_value) for scaling
        self.log_min = math.log(min_value)
        self.log_max = math.log(max_value)

    def init_params(self):
        """
        Initialize embedding parameters using Xavier uniform initialization.
        """
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass to compute the log-scaled embedding.
        
        Args:
            x (torch.Tensor): Input tensor of shape (B, T) containing pitch or amplitude values.
        
        Returns:
            torch.Tensor: Log-scaled embeddings of shape (B, T, embedding_dim).
        """
        # Tách riêng mask của các khung vô thanh (x <= 0)
        unvoiced_mask = (x <= 0)
        
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
        bin_indices[unvoiced_mask] = 0
        
        # Lấy embedding từ bin indices
        embeddings = self.embedding(bin_indices)
        return embeddings