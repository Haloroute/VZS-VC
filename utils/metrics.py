# MCD, F0 RMSE, SECS calculation functions
import torch
from torch import Tensor


# Function to calculate Accuracy for classification tasks
def calculate_accuracy(prediction: Tensor, target: Tensor, ignore_index: int = -100) -> tuple[int, int]:
    """
    Calculates the exact-match accuracy of token predictions. 
    A token is considered correct ONLY if all of its sub-tokens across D_codec match the target.

    Args:
        prediction (Tensor): The predicted logits (N, N_bins, T, D_codec).
        target (Tensor): The true labels (N, T, D_codec).
        ignore_index (int, optional): The index to ignore in the target. Defaults to -100.
    
    Returns:
        tuple[int, int]: A tuple containing the number of exactly correct tokens and the total number of valid tokens.
    """
    # Lấy class có xác suất cao nhất (N, T, D_codec)
    predicted_classes = torch.argmax(prediction, dim=1)

    # Khớp từng phần tử: shape (N, T, D_codec)
    element_match = (predicted_classes == target)

    # Khớp toàn bộ token: Gom nhóm theo D_codec (dim=-1), yêu cầu tất cả đều True -> shape (N, T)
    token_match = element_match.all(dim=-1)

    # Tạo mask bỏ qua ignore_index: Nếu token hợp lệ, không có phần tử nào là ignore_index -> shape (N, T)
    valid_mask = (target != ignore_index).all(dim=-1)

    # Đếm số lượng đúng và tổng số token hợp lệ
    correct_predictions = token_match.logical_and(valid_mask).sum().item()
    total_predictions = valid_mask.sum().item()

    return correct_predictions, total_predictions