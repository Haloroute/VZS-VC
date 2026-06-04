# MCD, F0 RMSE, SECS calculation functions
import torch
from torch import Tensor


# Function to calculate Accuracy for classification tasks
def calculate_accuracy(prediction: Tensor, target: Tensor, ignore_index: int = -100) -> tuple[int, int]:
    """
    Calculates the exact-match accuracy of token predictions. 
    A token is considered correct ONLY if all of its sub-tokens across D_codec match the target.

    Args:
        prediction (Tensor): The predicted logits (N, N_bins, T, D_fsq).
        target (Tensor): The true labels (N, T, D_fsq).
        ignore_index (int, optional): The index to ignore in the target. Defaults to -100.

    Returns:
        tuple[int, int]: A tuple containing the number of exactly correct tokens and the total number of valid tokens.
    """
    # Get the predicted classes by taking the argmax across the N_bins dimension (dim=1)
    predicted_classes = torch.argmax(prediction, dim=1) # (N, T, D_fsq)

    # Compare predicted classes with target: shape (N, T, D_fsq), True where they match
    element_match = (predicted_classes == target) # (N, T, D_fsq)

    # A token is correct if all its sub-tokens match, so we check if all elements along the D_fsq dimension match
    token_match = element_match.all(dim=-1) # (N, T) - True where all sub-tokens match for a token

    # Create a mask to ignore tokens where any sub-token is equal to the ignore_index
    valid_mask = (target != ignore_index).all(dim=-1) # (N, T) - True where all sub-tokens are valid (not ignore_index)

    # Calculate the number of correct predictions and total valid predictions
    correct_predictions = token_match.logical_and(valid_mask).sum().item() # Count tokens that are correct and valid
    total_predictions = valid_mask.sum().item() # Count all valid tokens

    return correct_predictions, total_predictions