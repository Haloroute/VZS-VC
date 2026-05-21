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
        hop_size: int = 320,
        decode_mode: str = 'local_argmax',
        threshold: float = 0.006,
        f0_min: float = 32.7,
        f0_max: float = 1975.5
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
            audio (torch.Tensor): Input audio waveform tensor, shape (B, T, 1) where B is batch size and T is the number of time steps.
        
        Returns:
            torch.Tensor: Extracted F0 sequence, shape (B, T_out) where T_out is the interpolated target length.
        """
        audio_length = audio.shape[1]
        f0_target_length = (audio_length // self.hop_size) + 1
        peak = torch.max(torch.abs(audio)) # Get the maximum absolute value in the audio tensor
        if peak > 1:
            audio = audio / (peak + 1e-8) # Normalize the audio to be in the range [-1, 1]

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

    def inference(self, x: torch.Tensor):
        """
        Perform inference using the FCPE model.
        Args:
            x: A 3D tensor (with batch dimension) containing the audio data in mono, 16kHz format (shape: (N, 1, T)).
        Returns:
            The output of the model after inference (shape: (N, T_pitch) with T_pitch being the number of time steps).
        """
        with torch.inference_mode():
            x = x.permute(0, 2, 1)  # (N, T, 1)
            return self.forward(x).squeeze(-1) # (N, T_pitch)