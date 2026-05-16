# Configuration class for the TTS model and training process
from dataclasses import dataclass

@dataclass
class VieNeuTTSDatasetConfig:
    # name: str = "pnnbao-ump/VieNeu-TTS-140h"  # Default dataset name
    path: str = "pnnbao-ump/VieNeu-TTS-140h"  # Default dataset path
    # sample_rate: int = 24000  # Sample rate for audio
    split: str = "train"  # VieNeu-TTS-140h has only 'train' split, so we will handle val/test splitting ourselves
    audio_column: str = "audio"  # Column name in the dataset that contains the audio data
    # seed: int = 42 # Random seed for reproducibility when loading the dataset

@dataclass
class VieNeuTTSPerturbationConfig:
    sample_rate: int = 16000  # Sample rate for output audio
    pitch_shift_down_range: tuple = (-6.5, -3.5)  # Pitch shift downward range (up to 6.5 semitones)
    pitch_shift_up_range: tuple = (-6.5, -3.5)  # Pitch shift upward range (up to 6.5 semitones)
    formant_shift_down_range: tuple = (1.10, 1.20)  # Formant shift down to simulate older/deeper voice
    formant_shift_up_range: tuple = (0.80, 0.90)  # Formant shift up to simulate younger/higher voice
    f0_high_thresh: float = 200.0  # High frequency threshold for F0
    f0_low_thresh: float = 110.0  # Low frequency threshold for F0
    eq_center_freq_range: tuple = (100.0, 8000.0)  # Range for equalizer center frequency
    eq_gain_range: tuple = (-12.0, 12.0)  # Range for equalizer gain adjustments in dB
    eq_q_range: tuple = (1.5, 4.0)  # Range for equalizer Q factor
    seed: int = 42 # Random seed for reproducibility when applying perturbations

@dataclass
class VieNeuTTSPerturbedDatasetConfig:
    name: str = "Haloroute/VieNeu-TTS-140h-perturbed"  # Default dataset name
    train_size: float = 0.9 # Proportion of data to use for training (rest will be used for validation)
    perturbed_audio_column: str = "perturbed_audio" # Column name for the perturbed audio in the new dataset
    seed: int = 42 # Random seed for reproducibility when splitting the dataset
