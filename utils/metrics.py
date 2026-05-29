# MCD, F0 RMSE, SECS calculation functions
import torch
from torch import Tensor


# Function to calculate Accuracy for classification tasks
def calculate_accuracy(prediction: Tensor, target: Tensor, ignore_index: int = -100) -> tuple[int, int]:
    """
    Calculates the accuracy of predictions against targets.

    Args:
        prediction (Tensor): The predicted labels (N, N_bins, T, D_codec).
        target (Tensor): The true labels (N, T, D_codec).
        ignore_index (int, optional): The index to ignore in the target. Defaults to -100.
    
    Returns:
        tuple[int, int]: A tuple containing the number of correct predictions and the total number of predictions.
    """
    # Find the predicted class for each sample
    predicted_classes = torch.argmax(prediction, dim=1)

    # Compare with the true labels
    mask = (target != ignore_index)
    correct_predictions = (predicted_classes == target).logical_and(mask).sum().item()
    total_predictions = mask.sum().item()

    return correct_predictions, total_predictions