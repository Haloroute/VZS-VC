# Configuration class for the modules used in the VC system
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

# Configuration for the DistillNeuCodec model used as the distilled codec module (a smaller version of NeuCodec for faster inference)
@dataclass
class DistillNeuCodecModuleConfig:
    pretrained_model_name_or_path: str = 'neuphonic/distill-neucodec'

# Configuration for the Voice Generator model used as the main VC model
@dataclass
class VoiceGeneratorModuleConfig:
    d_content: int = 512 # The dimensionality of the content embedding (came from VietASR content features). Should be 512.
    d_pitch: int = 32 # The dimensionality of the pitch embedding (after logarithmic embedding).
    d_amplitude: int = 32 # The dimensionality of the amplitude embedding (after logarithmic embedding).
    d_timbre: int = 192 # The dimensionality of the timbre embedding (came from ERes2NetV2). Should be 192.
    d_embedding: int = 1024 # The dimensionality of each token embedding. Should be 1024.
    n_tokens: int = 65538 # The number of input and output tokens (derived from NeuCodec codebook). Should be 2^16 + 2.

    n_pitch: int = 256 # The number of bins for pitch embedding.
    min_pitch: float = 32.7 # The minimum value for pitch embedding (should be a positive value). Should be around 32.7 (C1 note).
    max_pitch: float = 1244.5 # The maximum value for pitch embedding (should be a positive value). Should be around 1244.5 (D#5 note).
    n_amplitude: int = 256 # The number of bins for amplitude embedding.
    min_amplitude: float = 0.01 # The minimum value for amplitude embedding (should be a positive value). Should be around 0.01.
    max_amplitude: float = 0.85 # The maximum value for amplitude embedding (should be a positive value). Should be around 0.85.

    d_model: int = 512 # The dimensionality of the model (feature dimension).
    n_heads: int = 8 # The number of attention heads in each DiT block.
    d_ff: int = 1536 # The dimensionality of the feed-forward layer in each DiT block.
    n_layers: int = 8 # The number of DiT blocks in the generator.
    dropout: float = 0.2 # The dropout rate for regularization.