# Configuration class for the datasets used in the VC system
from dataclasses import dataclass

# Configuration for the original VieNeu-TTS-140h dataset
@dataclass
class VieNeuTTSDatasetConfig:
    # name: str = "pnnbao-ump/VieNeu-TTS-140h"  # Default dataset name
    path: str = "pnnbao-ump/VieNeu-TTS-140h"  # Default dataset path
    # sampling_rate: int = 24000  # Sampling rate for audio
    split: str = "train"  # VieNeu-TTS-140h has only 'train' split, so we will handle val/test splitting ourselves
    audio_column: str = "audio"  # Column name in the dataset that contains the audio data
    # seed: int = 42 # Random seed for reproducibility when loading the dataset

# Configuration for the DSP-based perturbation process
@dataclass
class VieNeuTTSPerturbationConfig:
    sampling_rate: int = 16000  # Sampling rate for output audio
    pitch_shift_down_range: tuple = (-6.5, -3.5)  # Pitch shift downward range (up to 6.5 semitones)
    pitch_shift_up_range: tuple = (3.5, 6.5)  # Pitch shift upward range (up to 6.5 semitones)
    formant_shift_down_range: tuple = (1.10, 1.20)  # Formant shift down to simulate older/deeper voice
    formant_shift_up_range: tuple = (0.80, 0.90)  # Formant shift up to simulate younger/higher voice
    f0_high_thresh: float = 200.0  # High frequency threshold for F0
    f0_low_thresh: float = 110.0  # Low frequency threshold for F0
    eq_center_freq_range: tuple = (100.0, 8000.0)  # Range for equalizer center frequency
    eq_gain_range: tuple = (-12.0, 12.0)  # Range for equalizer gain adjustments in dB
    eq_q_range: tuple = (1.5, 4.0)  # Range for equalizer Q factor
    seed: int = 42 # Random seed for reproducibility when applying perturbations

# Configuration for the new dataset created after applying DSP-based perturbation to VieNeu-TTS-140h
@dataclass
class VieNeuTTSPerturbedDatasetConfig:
    path: str = "Haloroute/VieNeu-TTS-140h-perturbed"  # Default dataset name
    train_size: float = 0.95 # Proportion of data to use for training (rest will be used for validation)
    perturbed_audio_column: str = "perturbed_audio" # Column name for the perturbed audio in the new dataset
    seed: int = 42 # Random seed for reproducibility when splitting the dataset

@dataclass
class VieNeuTTSPreprocessedDatasetConfig:
    path: str = "Haloroute/VieNeu-TTS-140h-preprocessed"
    train_split: str = "train" # Split name for training data in the preprocessed dataset
    val_split: str = "test" # Split name for validation data in the preprocessed dataset
    streaming: bool = False # Whether to load the dataset in streaming mode (useful for large datasets that don't fit in memory)

    amplitude_column: str = "amplitude_embedding" # Column name for amplitude embedding in the preprocessed dataset
    content_column: str = "content_embedding" # Column name for content embedding in the preprocessed dataset
    pitch_column: str = "pitch_embedding" # Column name for pitch embedding in the preprocessed dataset
    timbre_column: str = "timbre_embedding" # Column name for timbre embedding in the preprocessed dataset
    audio_column: str = "audio" # Column name for raw audio in the preprocessed dataset

    seed: int = 42 # Random seed for reproducibility when loading the preprocessed dataset