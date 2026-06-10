# Discriminator for VZS-VC
import torch.nn as nn

from torch import Tensor
from torch.nn.utils.parametrizations import weight_norm


# Discriminator module for VZS-VC
class VoiceDiscriminator(nn.Module):
    def __init__(
        self,
        d_model: int, n_layers: int,
        dropout: float = 0.2, n_mel_bins: int = 100
    ):
        super().__init__()
        # Configuration for the discriminator model
        self.d_model = d_model
        self.n_layers = n_layers
        self.dropout = dropout
        self.n_mel_bins = n_mel_bins

        # Define the convolutional layers of the discriminator model
        self.layers = nn.ModuleList()
        
        in_channels = n_mel_bins
        for i in range(n_layers):
            # Halve the length only on the very first layer
            stride = 2 if i == 0 else 1
            kernel_size = 4 if i == 0 else 3
            padding = 1 # Preserves length perfectly with k=3, s=1; Halves perfectly with k=4, s=2
            
            self.layers.append(
                nn.Sequential(
                    weight_norm(nn.Conv1d(
                        in_channels, d_model, 
                        kernel_size=kernel_size, stride=stride, padding=padding
                    )),
                    nn.LeakyReLU(0.2),
                    nn.Dropout(dropout)
                )
            )
            in_channels = d_model

        # Final pointwise convolution to score each frame independently
        self.final_layer = weight_norm(nn.Conv1d(d_model, 1, kernel_size=1))

        self.init_weights()

    def init_weights(self, std: float = 0.02):
        """
        Initialize the weights of the discriminator model using a normal distribution.
        """
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.normal_(m.weight, mean=0.0, std=std)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: Tensor) -> tuple[Tensor, list[Tensor]]:
        """Forward pass of the discriminator model

        Args:
            x (Tensor): Input Mel-spectrogram features of shape (N, 2T, n_mel_bins)

        Returns:
            tuple[Tensor, list[Tensor]]: A tuple containing:
            - out (Tensor): Validity scores of shape (N, T). Perfectly matches mask indices.
            - fmaps (list[Tensor]): Intermediate layer outputs for Feature Matching Loss.
                                  All feature maps from index 0 onwards will have temporal length T.
        """
        # Transpose from (N, 2T, n_mel_bins) to (N, n_mel_bins, 2T) for Conv1d
        x = x.transpose(1, 2) # (N, n_mel_bins, 2T)

        fmaps = []
        for layer in self.layers:
            x = layer(x) # (N, d_model, T) from the first layer onwards
            fmaps.append(x)

        out = self.final_layer(x) # (N, 1, T)
        fmaps.append(out)

        # Squeeze channel dimension to output shape (N, T)
        out = out.squeeze(1) # (N, T)

        return out, fmaps # (N, T), list of (N, d_model, T) for intermediate layers and (N, 1, T) for final layer