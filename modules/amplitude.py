# Use Local RMS for amplitude calculation.
import torch

import torch.nn as nn
import torch.nn.functional as F


# Local RMS amplitude extraction module.
class LocalRMSAmplitude(nn.Module):
    """
    Trích xuất đường viền cường độ âm thanh (Amplitude/Energy Contour) cục bộ
    bằng phương pháp Local RMS với cửa sổ trượt.
    """
    def __init__(self, window_size: int = 256, hop_size: int = 160, eps: float = 1e-8):
        """
        Args:
            window_size (int): Kích thước cửa sổ tính toán (số sample). Càng lớn đường bao càng trơn.
            hop_size (int): Khoảng cách giữa các cửa sổ. Càng nhỏ đường bao càng chi tiết.
            eps (float): Giá trị nhỏ để tránh lỗi chia cho 0 hoặc căn bậc hai của 0.
        """
        super(LocalRMSAmplitude, self).__init__()
        self.window_size = window_size
        self.hop_size = hop_size
        self.eps = eps

        # Tính toán padding để đảm bảo độ dài T không đổi
        self.pad_left = window_size // 2
        self.pad_right = window_size - self.pad_left

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x (torch.Tensor): Tensor âm thanh đầu vào có kích thước (B, T).
            
        Returns:
            torch.Tensor: Tensor chứa đường viền cường độ cục bộ, kích thước (B, T // hop_size + 1).
        """
        # Thêm chiều channel: (B, T) -> (B, 1, T)
        x = x.unsqueeze(1)
        
        # 1. Bình phương tín hiệu
        x_sq = x ** 2
        
        # 2. Padding hai đầu (dùng reflect để tránh sụt giảm năng lượng đột ngột ở viền)
        x_pad = F.pad(x_sq, (self.pad_left, self.pad_right), mode='reflect')
        
        # 3. Tính trung bình bình phương (Mean Square) bằng trượt cửa sổ
        mean_sq = F.avg_pool1d(x_pad, kernel_size=self.window_size, stride=self.hop_size, padding=0)
        
        # 4. Tính căn bậc hai (Root) để thu được RMS
        rms = torch.sqrt(mean_sq + self.eps)
        
        # Loại bỏ chiều channel: (B, 1, T) -> (B, T)
        return rms.squeeze(1)