# Preprocessing scripts for audio data
import datasets, os, torch, torchaudio
import numpy as np

from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from datasets import Audio, Dataset, DatasetDict, Features
from numpy import ndarray
from torch import Tensor
from torchaudio.transforms import Resample
from torchcodec import AudioSamples

# from data.dataset import VieNeuTTSDataset
from data.preprocess import dsp_perturbate
from modules import (
    EncoderModel,
    ERes2NetV2,
    FCPE,
    LocalRMSAmplitude,
    NeuCodec
)
from utils.configs import (
    VieNeuTTSDatasetConfig,
    VieNeuTTSPerturbationConfig,
    VieNeuTTSPerturbedDatasetConfig,
    VieNeuTTSPreprocessedDatasetConfig
)
from utils.modules import (
    load_amplitude_encoder,
    load_codec,
    load_content_encoder,
    load_pitch_encoder,
    load_timbre_encoder
)


# Create a new dataset with DSP-based perturbation applied to the audio.
def audio_perturbate():
    # Load configurations for the dataset and perturbation from the YAML config file.
    # dataset_config = yaml.safe_load(open("configs/dataset_config.yaml", encoding="utf-8"))
    vieneu_tts_dataset_config = VieNeuTTSDatasetConfig()
    vieneu_tts_perturbation_config = VieNeuTTSPerturbationConfig()
    vieneu_tts_perturbed_dataset_config = VieNeuTTSPerturbedDatasetConfig()

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

    # Define a function to apply DSP-based perturbation to each audio sample in the dataset. This function will be applied to each sample in the dataset using the map function.
    def dsp_perturbation_fn(sample: dict, idx: int) -> dict:
        try:
            torch.set_num_threads(1)
            audio: AudioSamples = sample.get(vieneu_tts_dataset_config.audio_column, None).get_all_samples()
            sampling_rate: int = audio.sample_rate
            audio_tensor: Tensor = audio.data # (C, T)

            # Resample if needed
            if audio and sampling_rate != vieneu_tts_perturbation_config.sampling_rate:
                if sampling_rate not in resampler_dict:
                    resampler_dict[sampling_rate] = torchaudio.transforms.Resample(orig_freq=sampling_rate, new_freq=vieneu_tts_perturbation_config.sampling_rate)
                resampler = resampler_dict[sampling_rate]
                audio_tensor: Tensor = resampler(audio_tensor) # (C, T')

            # Apply DSP-based perturbation to the audio in the batch.
            perturbed_audio: ndarray = dsp_perturbate(
                audio_np=audio_tensor.squeeze(0).numpy(), # Convert to numpy array 
                **asdict(vieneu_tts_perturbation_config) # Unpack perturbation config parameters
            )
            sample[vieneu_tts_perturbed_dataset_config.perturbed_audio_column] = {
                "array": perturbed_audio.astype(np.float32), # Store perturbed audio as float32 numpy array
                "sampling_rate": vieneu_tts_perturbation_config.sampling_rate
            }
            return sample
        except Exception as e:
            print(f"Error processing sample at index {idx} during DSP perturbation! Error: {e}")
            raise e # Reraise the exception to be handled by the map function's error handling

    # Create a new dataset with the perturbed audio and the same features as the original dataset, plus a new feature for the perturbed audio.
    vieneu_tts_perturbed_dataset = vieneu_tts_dataset.map(
        dsp_perturbation_fn,
        with_indices=True,
        features=Features({
            **vieneu_tts_dataset.features, # Original features
            "perturbed_audio": Audio(decode=True) # New feature for perturbed audio
        }),
        num_proc=max(1, os.cpu_count()), # Use multiple processes for faster splitting if possible
        desc="Applying DSP-based perturbation"
    )

    # Split the perturbed dataset into training and validation sets based on the specified train split ratio.
    vieneu_tts_perturbed_dataset = vieneu_tts_perturbed_dataset.train_test_split(
        train_size=vieneu_tts_perturbed_dataset_config.train_size,
        seed=vieneu_tts_perturbed_dataset_config.seed
    )
    # Upload the perturbed dataset to Hugging Face Hub with the specified path.
    vieneu_tts_perturbed_dataset.push_to_hub(vieneu_tts_perturbed_dataset_config.path)
    print(f"Perturbed dataset created and pushed to Hugging Face Hub with name: {vieneu_tts_perturbed_dataset_config.path}")


# Extract embeddings from the perturbed dataset and save it to a new dataset
def extract_embeddings():
    # Load configurations for the dataset and perturbation from the YAML config file.
    # dataset_config = yaml.safe_load(open("configs/dataset_config.yaml", encoding="utf-8"))
    vieneu_tts_dataset_config = VieNeuTTSDatasetConfig()
    vieneu_tts_perturbation_config = VieNeuTTSPerturbationConfig()
    vieneu_tts_perturbed_dataset_config = VieNeuTTSPerturbedDatasetConfig()
    vieneu_tts_preprocessed_dataset_config = VieNeuTTSPreprocessedDatasetConfig()

    # Load the perturbed datasets from Hugging Face Hub using the specified configuration.
    vieneu_tts_perturbed_datasets: DatasetDict = datasets.load_dataset(vieneu_tts_perturbed_dataset_config.path)

    # Load the pre-trained modules
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amplitude_encoder: LocalRMSAmplitude = load_amplitude_encoder(device)
    codec: NeuCodec = load_codec(device)
    content_encoder: EncoderModel = load_content_encoder(device)
    pitch_encoder: FCPE = load_pitch_encoder(device)
    timbre_encoder: ERes2NetV2 = load_timbre_encoder(device)
    resampler_dict = {} # Store resampler to avoid re-initialization for each sample if needed

    # Initialize multi-processing operations
    streams = [torch.cuda.Stream() for _ in range(5)] # Create separate streams for each encoder to allow parallel processing
    executor = ThreadPoolExecutor(max_workers=5) # Use a thread pool executor to run the encoders in parallel

    def inference_fn(func, input_tensor, stream, *args):
        with torch.cuda.stream(stream), torch.inference_mode():
            return func(input_tensor, *args)

    # Define a function to extract embeddings from each sample in the perturbed dataset. This function will be applied to each sample in the dataset using the map function.
    def embedding_extraction_fn(sample: dict, idx: int) -> dict:
        try:
            # Extract the original and perturbed audio from the sample
            # torch.set_num_threads(1)
            orignal_audio: AudioSamples = sample.get(vieneu_tts_dataset_config.audio_column, None).get_all_samples()
            perturbed_audio: AudioSamples = sample.get(vieneu_tts_perturbed_dataset_config.perturbed_audio_column, None).get_all_samples()
            original_sampling_rate: int = orignal_audio.sample_rate
            perturbed_sampling_rate: int = perturbed_audio.sample_rate
            original_audio_tensor: Tensor = orignal_audio.data.unsqueeze(0).to(device, non_blocking=True) # (1, C, T_24k)
            perturbed_audio_tensor: Tensor = perturbed_audio.data.unsqueeze(0).to(device, non_blocking=True) # (1, C, T_16k)

            # Resample if needed
            if original_sampling_rate != vieneu_tts_perturbation_config.sampling_rate:
                if original_sampling_rate not in resampler_dict:
                    resampler_dict[original_sampling_rate] = Resample(orig_freq=original_sampling_rate, new_freq=vieneu_tts_perturbation_config.sampling_rate).to(device)
                resampler = resampler_dict[original_sampling_rate]
                original_audio_tensor: Tensor = resampler(original_audio_tensor) # (1, C, T')

            if perturbed_sampling_rate != vieneu_tts_perturbation_config.sampling_rate:
                if perturbed_sampling_rate not in resampler_dict:
                    resampler_dict[perturbed_sampling_rate] = Resample(orig_freq=perturbed_sampling_rate, new_freq=vieneu_tts_perturbation_config.sampling_rate).to(device)
                resampler = resampler_dict[perturbed_sampling_rate]
                perturbed_audio_tensor: Tensor = resampler(perturbed_audio_tensor) # (1, C, T')

            # Create funtions to run each encoder in parallel
            f_amplitude = executor.submit(inference_fn, amplitude_encoder.inference, original_audio_tensor, streams[0])
            f_content = executor.submit(inference_fn, content_encoder.inference, perturbed_audio_tensor, streams[1])
            f_pitch = executor.submit(inference_fn, pitch_encoder.inference, original_audio_tensor, streams[2])
            f_timbre = executor.submit(inference_fn, timbre_encoder.inference, original_audio_tensor, streams[3])
            f_codec = executor.submit(inference_fn, codec.encode_pre_vq, original_audio_tensor, streams[4])

            # Wait for all encoders to finish and get the results
            torch.cuda.synchronize() # Ensure all streams have finished processing
            amplitude_embedding: Tensor = f_amplitude.result() # (1, T)
            content_embedding: Tensor = f_content.result() # (1, T, D_content)
            pitch_embedding: Tensor = f_pitch.result() # (1, T)
            timbre_embedding: Tensor = f_timbre.result() # (1, D_timbre)
            pre_vq_embedding, acoustic_embedding = f_codec.result() # (1, T, D_pre_vq), (1, T, D_acoustic)

            # Store the extracted embeddings in the sample dictionary
            del sample[vieneu_tts_dataset_config.audio_column], sample[vieneu_tts_perturbed_dataset_config.perturbed_audio_column] # Remove original and perturbed audio from the sample to save space, we only keep the embeddings
            sample[vieneu_tts_preprocessed_dataset_config.amplitude_column] = amplitude_embedding.squeeze(0).numpy(force=True)
            sample[vieneu_tts_preprocessed_dataset_config.content_column] = content_embedding.squeeze(0).numpy(force=True)
            sample[vieneu_tts_preprocessed_dataset_config.pitch_column] = pitch_embedding.squeeze(0).numpy(force=True)
            sample[vieneu_tts_preprocessed_dataset_config.timbre_column] = timbre_embedding.squeeze(0).numpy(force=True)
            sample[vieneu_tts_preprocessed_dataset_config.pre_vq_column] = pre_vq_embedding.squeeze(0).numpy(force=True)
            sample[vieneu_tts_preprocessed_dataset_config.acoustic_column] = acoustic_embedding.squeeze(0).numpy(force=True)

            return sample
        except Exception as e:
            print(f"Error processing sample at index {idx} during embedding extraction! Error: {e}")
            raise e # Reraise the exception to be handled by the map function's error handling

    # Create a new dataset with the extracted embeddings and the same features as the perturbed dataset, plus new features for the embeddings.
    vieneu_tts_preprocessed_dataset: DatasetDict = vieneu_tts_perturbed_datasets.map(
        embedding_extraction_fn,
        with_indices=True,
        num_proc=0,
        desc="Extracting embeddings from perturbed dataset"
    )

    # Upload the preprocessed dataset with embeddings to Hugging Face Hub
    vieneu_tts_preprocessed_dataset.push_to_hub(vieneu_tts_preprocessed_dataset_config.path)
    print(f"Perturbed dataset created and pushed to Hugging Face Hub with name: {vieneu_tts_perturbed_dataset_config.path}")


def main():
    print("This is the main preprocessing script. You can run specific preprocessing functions from here if needed.")
    print("Choose an action:")
    print("1. Apply DSP-based perturbation to audio data")
    print("2. Extract input embeddings using pre-trained models")
    choice = input("Enter the number of your choice: ")
    if choice == "1":
        print("You chose to apply DSP-based perturbation.")
        audio_perturbate()
    elif choice == "2":
        print("You chose to extract input embeddings.")
        extract_embeddings()
    else:
        print("Invalid choice. Please run the script again and choose a valid option.")

if __name__ == "__main__":
    main()