# Preprocessing scripts for audio data
import datasets, os, torch, torchaudio, yaml
import numpy as np

from dataclasses import asdict
from datasets import Audio, Dataset, Features
from numpy import ndarray
from torch import Tensor
from torchcodec import AudioSamples

# from data.dataset import VieNeuTTSDataset
from data.preprocess import dsp_perturbate
from utils.configs import VieNeuTTSDatasetConfig, VieNeuTTSPerturbationConfig, VieNeuTTSPerturbedDatasetConfig


# Create a new dataset with DSP-based perturbation applied to the audio.
def audio_perturbate():
    # Load configurations for the dataset and perturbation from the YAML config file.
    dataset_config = yaml.safe_load(open("configs/dataset_config.yaml", encoding="utf-8"))
    vieneu_tts_dataset_config = VieNeuTTSDatasetConfig(**dataset_config["vieneu_tts_dataset"])
    vieneu_tts_perturbation_config = VieNeuTTSPerturbationConfig(**dataset_config["vieneu_tts_perturbation"])
    vieneu_tts_perturbed_dataset_config = VieNeuTTSPerturbedDatasetConfig(**dataset_config["vieneu_tts_perturbated_dataset"])

    # Load the original VieNeu-TTS-140h dataset using the specified configuration.
    # vieneu_tts_dataset = VieNeuTTSDataset(
    #     **asdict(vieneu_tts_dataset_config),
    #     part="train", # We will handle train/val splitting ourselves after perturbation,
    #     val_size=0.0, # No validation split at this stage, we will split after perturbation
    # )
    vieneu_tts_dataset: Dataset = datasets.load_dataset(
        vieneu_tts_dataset_config.path,
        split=vieneu_tts_dataset_config.split
    )
    vieneu_tts_dataset = vieneu_tts_dataset.cast_column(vieneu_tts_dataset_config.audio_column, Audio(decode=True)) # Ensure audio column is decoded to (array, sampling_rate) format
    resampler_dict = {} # Store resampler to avoid re-initialization for each sample if needed

    def dsp_perturbation_fn(sample: dict) -> dict:
        torch.set_num_threads(1)
        audio: AudioSamples = sample.get(vieneu_tts_dataset_config.audio_column, None).get_all_samples()
        orig_sr: int = audio.sample_rate
        audio_tensor: Tensor = audio.data # (C, T)
        # audio_bytes: bytes = sample.get("audio").get("bytes")
        # audio_tensor, orig_sr = torchaudio.load(io.BytesIO(audio_bytes)) # (C, T)

        # Resample if needed
        if audio and orig_sr != vieneu_tts_perturbation_config.sample_rate:
            if orig_sr not in resampler_dict:
                resampler_dict[orig_sr] = torchaudio.transforms.Resample(orig_freq=orig_sr, new_freq=vieneu_tts_perturbation_config.sample_rate)
            resampler = resampler_dict[orig_sr]
            audio_tensor: Tensor = resampler(audio_tensor) # (C, T')

        # Apply DSP-based perturbation to the audio in the batch.
        perturbed_audio: ndarray = dsp_perturbate(
            audio_np=audio_tensor.squeeze(0).numpy(), # Convert to numpy array 
            **asdict(vieneu_tts_perturbation_config) # Unpack perturbation config parameters
        )
        sample[vieneu_tts_perturbed_dataset_config.perturbed_audio_column] = {
            "array": perturbed_audio.astype(np.float32), # Store perturbed audio as float32 numpy array
            "sampling_rate": vieneu_tts_perturbation_config.sample_rate
        }
        return sample
    
    # Create a new dataset with the perturbed audio and the same features as the original dataset, plus a new feature for the perturbed audio.
    vieneu_tts_perturbed_dataset = vieneu_tts_dataset.map(
        dsp_perturbation_fn,
        features=Features({
            **vieneu_tts_dataset.features.to_dict(), # Original features
            "perturbed_audio": Audio(decode=False) # New feature for perturbed audio
        }),
        num_proc=max(1, os.cpu_count() - 1), # Use multiple processes for faster splitting if possible
        desc="Applying DSP-based perturbation"
    )

    # Split the perturbed dataset into training and validation sets based on the specified train split ratio.
    vieneu_tts_perturbed_dataset = vieneu_tts_perturbed_dataset.train_test_split(
        train_size=vieneu_tts_perturbed_dataset_config.train_size,
        seed=vieneu_tts_perturbed_dataset_config.seed
    )
    # Upload the perturbed dataset to Hugging Face Hub with the specified name.
    vieneu_tts_perturbed_dataset.push_to_hub(vieneu_tts_perturbed_dataset_config.name)
    print(f"Perturbed dataset created and pushed to Hugging Face Hub with name: {vieneu_tts_perturbed_dataset_config.name}")

def main():
    print("This is the main preprocessing script. You can run specific preprocessing functions from here if needed.")
    print("Choose an action:")
    print("1. Apply DSP-based perturbation to audio data")
    print("2. Extract content embeddings using a VietASR encoder model")
    print("3. Extract speaker embeddings using a ERes2Net-V2 model")
    print("4. Extract F0 contours using FCPE model")
    choice = input("Enter the number of your choice: ")
    if choice == "1":
        print("You chose to apply DSP-based perturbation.")
        audio_perturbate()
    elif choice == "2":
        print("You chose to extract content embeddings. Please use the VietASR encoder model for this task.")
    elif choice == "3":
        print("You chose to extract speaker embeddings. Please use the ERes2Net-V2 model for this task.")
    elif choice == "4":
        print("You chose to extract F0 contours. Please use the FCPE model for this task.")
    else:
        print("Invalid choice. Please run the script again and choose a valid option.")

if __name__ == "__main__":
    main()