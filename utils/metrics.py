# MCD, F0 RMSE, SECS calculation functions
import torch
from torch import Tensor


# Function to calculate Accuracy for classification tasks
def calculate_accuracy(prediction: Tensor, target: Tensor, ignore_index: int = -100) -> tuple[int, int]:
    """
    Calculates the exact-match accuracy of token predictions. 

    Args:
        prediction (Tensor): The predicted logits (N, N_tokens, T).
        target (Tensor): The true labels (N, T).
        ignore_index (int, optional): The index to ignore in the target. Defaults to -100.
    
    Returns:
        tuple[int, int]: A tuple containing the number of correctly predicted tokens and the total number of valid tokens.
    """
    # Get the class with the highest probability -> (N, T)
    predicted_classes = torch.argmax(prediction, dim=1)

    # Create a mask for valid tokens (ignoring padding) -> (N, T)
    valid_mask = (target != ignore_index)

    # Count correct predictions only where the mask is valid
    correct_predictions = (predicted_classes == target).masked_select(valid_mask).sum().item()
    total_predictions = valid_mask.sum().item()

    return correct_predictions, total_predictions