# Use Local RMS for amplitude calculation.
import torch

import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor


# Local RMS amplitude extraction module.
class LocalRMSAmplitude(nn.Module):
    """
    Trích xuất đường viền cường độ âm thanh (Amplitude/Energy Contour) cục bộ
    bằng phương pháp Local RMS với cửa sổ trượt.
    """
    def __init__(self, window_size: int = 960, hop_size: int = 320):
        """
        Args:
            window_size (int): Kích thước cửa sổ tính toán (số sample). Càng lớn đường bao càng trơn.
            hop_size (int): Khoảng cách giữa các cửa sổ. Càng nhỏ đường bao càng chi tiết.
        """
        super(LocalRMSAmplitude, self).__init__()
        self.window_size = window_size
        self.hop_size = hop_size

        # Tính toán padding để đảm bảo độ dài T không đổi
        total_padding = window_size - hop_size
        self.pad_left = total_padding // 2
        self.pad_right = total_padding - self.pad_left

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass.

        Args:
            x (torch.Tensor): Tensor âm thanh đầu vào có kích thước (B, 1, T).

        Returns:
            torch.Tensor: Tensor chứa đường viền cường độ cục bộ, kích thước (B, T // hop_size).
        """
        # 1. Bình phương tín hiệu
        x_sq = x ** 2

        # 2. Padding hai đầu (dùng reflect để tránh sụt giảm năng lượng đột ngột ở viền)
        x_pad = F.pad(x_sq, (self.pad_left, self.pad_right), mode='reflect')

        # 3. Tính trung bình bình phương (Mean Square) bằng trượt cửa sổ
        mean_sq = F.avg_pool1d(x_pad, kernel_size=self.window_size, stride=self.hop_size, padding=0)

        # 4. Tính căn bậc hai (Root) để thu được RMS
        rms = torch.sqrt(mean_sq + 1e-8)  # Thêm epsilon nhỏ để tránh chia cho 0

        # Loại bỏ chiều channel: (B, 1, T) -> (B, T)
        return rms.squeeze(1)

    def inference(self, x: Tensor):
        """
        Perform inference using LocalRMS algorithm.

        Args:
            x: A 3D tensor (with batch dimension) containing the audio data in mono, 16kHz format (shape: (N, 1, T)).

        Returns:
            The output of the model after inference (shape: (N, T_amplitude) with T_amplitude being the number of time steps).
        """
        with torch.inference_mode():
            return self.forward(x) # (N, T_amplitude)