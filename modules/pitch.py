# Wrapper network for pitch extraction using TorchFCPE
import torch
import torch.nn as nn

from torchfcpe import spawn_bundled_infer_model


# Wrapper network for pitch extraction using TorchFCPE.
class FCPE(nn.Module):
    """
    Wrapper network for pitch extraction using TorchFCPE.
    """
    def __init__(
        self,
        sampling_rate: int = 16000,
        hop_size: int = 160,
        decode_mode: str = 'local_argmax',
        threshold: float = 0.006,
        f0_min: float = 80,
        f0_max: float = 880
    ):
        super(FCPE, self).__init__()
        
        # Load the TorchFCPE model
        self.model = spawn_bundled_infer_model(device='cpu')
        
        # Store configuration parameters
        self.sampling_rate = sampling_rate
        self.hop_size = hop_size
        self.decode_mode = decode_mode
        self.threshold = threshold
        self.f0_min = f0_min
        self.f0_max = f0_max

    def to(self, *args, **kwargs):
        self.model = self.model.to(*args, **kwargs)
        return self

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Extract pitch using TorchFCPE.
        
        Args:
            audio (torch.Tensor): Input audio waveform tensor, shape (B, T) or (1, T)
        
        Returns:
            torch.Tensor: Extracted F0 sequence, shape (B, T_out) where T_out is the interpolated target length.
        """
        audio_length = audio.shape[-1]
        f0_target_length = (audio_length // self.hop_size) + 1
        
        f0 = self.model.infer(
            audio,
            sr=self.sampling_rate,
            decoder_mode=self.decode_mode,
            threshold=self.threshold,
            f0_min=self.f0_min,
            f0_max=self.f0_max,
            interp_uv=False,
            output_interp_target_length=f0_target_length
        )
        return f0