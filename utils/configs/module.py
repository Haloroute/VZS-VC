# Configuration class for the TTS model and training process
from dataclasses import dataclass

# Configuration for the ERes2Net-V2 model used as the timbre encoder module
@dataclass
class ERes2NetV2ModuleConfig:
    checkpoint_path: str = 'checkpoints/timbre_encoder.safetensors'
    n_mel_bins: int = 80
    sampling_rate: int = 16000

# Configuration for the FCPE model used as the pitch encoder module
@dataclass
class FCPEModuleConfig:
    sampling_rate: int = 16000
    hop_size: int = 320
    f0_min: float = 32.7
    f0_max: float = 1975.5

# Configuration for the Zipformer2 model used as the content encoder module
@dataclass
class Zipformer2ModuleConfig:
    checkpoint_path: str = 'checkpoints/content_encoder.safetensors'
    dither: int = 0
    high_freq: int = -400
    n_mel_bins: int = 80
    sampling_rate: int = 16000
    snip_edges: bool = False

# Configuration for the LocalRMS method used as the amplitude encoder module
@dataclass
class LocalRMSModuleConfig:
    window_size: int = 960
    hop_size: int = 320

# Configuration for the NeuCodec model used as the codec module
@dataclass
class NeuCodecModuleConfig:
    pretrained_model_name_or_path: str = 'neuphonic/neucodec'
