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

# Configuration for the Voice Generator model used as the main VC model
@dataclass
class VoiceGeneratorModuleConfig:
    d_content: int = 512 # The dimensionality of the content embedding (came from VietASR content features). Should be 512.
    d_pitch: int = 32 # The dimensionality of the pitch embedding (after logarithmic embedding).
    d_amplitude: int = 32 # The dimensionality of the amplitude embedding (after logarithmic embedding).

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

    sample_rate: int = 24000 # The sampling rate for the input and output Mel-spectrograms (should match the sampling rate used for the Mel-spectrogram features in the dataset, which is 24000).
    n_fft: int = 1024 # The size of the FFT for computing the Mel-spectrogram.
    hop_length: int = 240 # The hop length (frame shift) for computing the Mel-spectrogram (should be chosen to achieve the desired temporal resolution, e.g., 240 for 100Hz at 24kHz).
    n_mel_bins: int = 100 # The number of Mel frequency bins in the intermediate spectrogram (should match the number of bins used for the Vocoder, which is 100).

# Configuration for the Discriminator model used as the discriminator in the VC system
@dataclass
class VoiceDiscriminatorModuleConfig:
    d_model: int = 512 # The dimensionality of the discriminator model.
    n_layers: int = 6 # The number of convolutional layers in the discriminator model.
    dropout: float = 0.2 # The dropout rate for regularization in the discriminator model.
    n_mel_bins: int = 100 # The number of Mel frequency bins in the input features for the discriminator (should match the n_mel_bins used in the generator and dataset).

# Configuration for the BigVGAN model used as the vocoder in the VC system
@dataclass
class BigVGANModuleConfig:
    pretrained_model_name_or_path: str = 'nvidia/bigvgan_v2_24khz_100band_256x' # The name or path of the pretrained BigVGAN model to use as the vocoder (should be compatible with the n_mel_bins used in the generator, which is 100).