# Preprocessing scripts for audio data
import datasets, os, torch, torchaudio
import numpy as np

from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from datasets import Array2D, Audio, Dataset, DatasetDict, Features, Sequence, Value
from numpy import ndarray
from torch import Tensor
from torchaudio.transforms import Resample
from torchcodec import AudioSamples

from modules import (
    EncoderModel,
    ERes2NetV2,
    FCPE,
    LocalRMSAmplitude,
    NeuCodec
)
from utils.audio import dsp_perturbate
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
    dataset_config = VieNeuTTSDatasetConfig()
    perturbation_config = VieNeuTTSPerturbationConfig()
    perturbed_dataset_config = VieNeuTTSPerturbedDatasetConfig()

    # Load the original VieNeu-TTS-140h dataset using the specified configuration.
    dataset: Dataset = datasets.load_dataset(
        dataset_config.path,
        split=dataset_config.split
    )
    dataset = dataset.cast_column(dataset_config.audio_column, Audio(decode=True)) # Ensure audio column is decoded to (array, sampling_rate) format
    resampler_dict = {} # Store resampler to avoid re-initialization for each sample if needed

    # Define a function to apply DSP-based perturbation to each audio sample in the dataset. This function will be applied to each sample in the dataset using the map function.
    def dsp_perturbation_fn(sample: dict, idx: int) -> dict:
        try:
            torch.set_num_threads(1)
            audio: AudioSamples = sample.get(dataset_config.audio_column, None).get_all_samples()
            sampling_rate: int = audio.sample_rate
            audio_tensor: Tensor = audio.data # (C, T)

            # Resample if needed
            if audio and sampling_rate != perturbation_config.sampling_rate:
                if sampling_rate not in resampler_dict:
                    resampler_dict[sampling_rate] = torchaudio.transforms.Resample(orig_freq=sampling_rate, new_freq=perturbation_config.sampling_rate)
                resampler = resampler_dict[sampling_rate]
                audio_tensor: Tensor = resampler(audio_tensor) # (C, T')

            # Apply DSP-based perturbation to the audio in the batch.
            perturbed_audio: ndarray = dsp_perturbate(
                audio_np=audio_tensor.squeeze(0).numpy(), # Convert to numpy array 
                **asdict(perturbation_config) # Unpack perturbation config parameters
            )
            sample[perturbed_dataset_config.perturbed_audio_column] = {
                "array": perturbed_audio.astype(np.float32), # Store perturbed audio as float32 numpy array
                "sampling_rate": perturbation_config.sampling_rate
            }
            return sample
        except Exception as e:
            print(f"Error processing sample at index {idx} during DSP perturbation! Error: {e}")
            raise e # Reraise the exception to be handled by the map function's error handling

    # Create a new dataset with the perturbed audio and the same features as the original dataset, plus a new feature for the perturbed audio.
    perturbed_dataset = dataset.map(
        dsp_perturbation_fn,
        with_indices=True,
        features=Features({
            **dataset.features, # Original features
            "perturbed_audio": Audio(decode=True) # New feature for perturbed audio
        }),
        num_proc=max(1, os.cpu_count()), # Use multiple processes for faster splitting if possible
        desc="Applying DSP-based perturbation"
    )

    # Split the perturbed dataset into training and validation sets based on the specified train split ratio.
    perturbed_dataset = perturbed_dataset.train_test_split(
        train_size=perturbed_dataset_config.train_size,
        seed=perturbed_dataset_config.seed
    )
    # Upload the perturbed dataset to Hugging Face Hub with the specified path.
    perturbed_dataset.push_to_hub(perturbed_dataset_config.path)
    print(f"Perturbed dataset created and pushed to Hugging Face Hub with name: {perturbed_dataset_config.path}")


# Extract embeddings from the perturbed dataset and save it to a new dataset
def extract_embeddings():
    # Load configurations for the dataset and perturbation from the YAML config file.
    # dataset_config = yaml.safe_load(open("configs/dataset_config.yaml", encoding="utf-8"))
    dataset_config = VieNeuTTSDatasetConfig()
    perturbation_config = VieNeuTTSPerturbationConfig()
    perturbed_dataset_config = VieNeuTTSPerturbedDatasetConfig()
    preprocessed_dataset_config = VieNeuTTSPreprocessedDatasetConfig()

    # Load the perturbed datasets from Hugging Face Hub using the specified configuration.
    perturbed_datasets: DatasetDict = datasets.load_dataset(perturbed_dataset_config.path)

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
            orignal_audio: AudioSamples = sample.get(dataset_config.audio_column, None).get_all_samples()
            perturbed_audio: AudioSamples = sample.get(perturbed_dataset_config.perturbed_audio_column, None).get_all_samples()
            original_sampling_rate: int = orignal_audio.sample_rate
            perturbed_sampling_rate: int = perturbed_audio.sample_rate
            original_audio_tensor: Tensor = orignal_audio.data.unsqueeze(0).to(device, non_blocking=True) # (1, C, T_24k)
            perturbed_audio_tensor: Tensor = perturbed_audio.data.unsqueeze(0).to(device, non_blocking=True) # (1, C, T_16k)

            # Resample if needed
            if original_sampling_rate != perturbation_config.sampling_rate:
                if original_sampling_rate not in resampler_dict:
                    resampler_dict[original_sampling_rate] = Resample(orig_freq=original_sampling_rate, new_freq=perturbation_config.sampling_rate).to(device)
                resampler = resampler_dict[original_sampling_rate]
                original_audio_tensor: Tensor = resampler(original_audio_tensor) # (1, C, T')

            if perturbed_sampling_rate != perturbation_config.sampling_rate:
                if perturbed_sampling_rate not in resampler_dict:
                    resampler_dict[perturbed_sampling_rate] = Resample(orig_freq=perturbed_sampling_rate, new_freq=perturbation_config.sampling_rate).to(device)
                resampler = resampler_dict[perturbed_sampling_rate]
                perturbed_audio_tensor: Tensor = resampler(perturbed_audio_tensor) # (1, C, T')

            # Create funtions to run each encoder in parallel
            f_amplitude = executor.submit(inference_fn, amplitude_encoder.inference, original_audio_tensor, streams[0])
            f_content = executor.submit(inference_fn, content_encoder.inference, perturbed_audio_tensor, streams[1])
            f_pitch = executor.submit(inference_fn, pitch_encoder.inference, original_audio_tensor, streams[2])
            f_timbre = executor.submit(inference_fn, timbre_encoder.inference, original_audio_tensor, streams[3])
            f_codec = executor.submit(inference_fn, codec.encode_code, original_audio_tensor, streams[4])

            # Wait for all encoders to finish and get the results
            amplitude_embedding: Tensor = f_amplitude.result() # (1, T)
            content_embedding: Tensor = f_content.result() # (1, T', D_content)
            pitch_embedding: Tensor = f_pitch.result() # (1, T)
            timbre_embedding: Tensor = f_timbre.result() # (1, D_timbre)
            code_embedding: Tensor = f_codec.result() # (1, T)
            torch.cuda.synchronize() # Ensure all streams have finished processing

            # Validate that the extracted embeddings do not contain NaN values before yielding the sample
            if amplitude_embedding.isnan().any():
                raise ValueError(f"Amplitude embedding contains NaN values for sample at index {idx}.")
            if content_embedding.isnan().any():
                raise ValueError(f"Content embedding contains NaN values for sample at index {idx}.")
            if pitch_embedding.isnan().any():
                raise ValueError(f"Pitch embedding contains NaN values for sample at index {idx}.")
            if timbre_embedding.isnan().any():
                raise ValueError(f"Timbre embedding contains NaN values for sample at index {idx}.")
            if code_embedding.isnan().any():
                raise ValueError(f"Code embedding contains NaN values for sample at index {idx}.")

            # Build the new yielded dictionary, excluding un-serializable audio columns
            new_sample = {
                k: v for k, v in sample.items() 
                if k not in [dataset_config.audio_column, perturbed_dataset_config.perturbed_audio_column]
            }

            # Store the extracted embeddings in the sample dictionary
            new_sample[preprocessed_dataset_config.amplitude_column] = amplitude_embedding.squeeze(0).numpy(force=True)
            new_sample[preprocessed_dataset_config.content_column] = content_embedding.squeeze(0).numpy(force=True)
            new_sample[preprocessed_dataset_config.pitch_column] = pitch_embedding.squeeze(0).numpy(force=True)
            new_sample[preprocessed_dataset_config.timbre_column] = timbre_embedding.squeeze(0).numpy(force=True)
            new_sample[preprocessed_dataset_config.code_column] = code_embedding.squeeze(0).numpy(force=True)

            return new_sample
        except Exception as e:
            print(f"Error processing sample at index {idx} during embedding extraction! Error: {e}")
            raise e # Reraise the exception to be handled by the map function's error handling

    # Define the features for the new dataset, which include the original features from the perturbed dataset plus new features for each of the extracted embeddings.
    preprocessed_features = Features({
        **{k: v for k, v in perturbed_datasets["train"].features.items() if k != perturbed_dataset_config.perturbed_audio_column and k != dataset_config.audio_column}, # Original features from perturbed dataset
        preprocessed_dataset_config.amplitude_column: Sequence(Value("float32")), # New feature for amplitude embedding
        preprocessed_dataset_config.content_column: Array2D(shape=(None, 512), dtype="float32"), # New feature for content embedding
        preprocessed_dataset_config.pitch_column: Sequence(Value("float32")), # New feature for pitch embedding
        preprocessed_dataset_config.timbre_column: Sequence(Value("float32"), length=192), # New feature for timbre embedding
        preprocessed_dataset_config.code_column: Sequence(Value("int64")), # New feature for code embedding
    })

    # Create a new dataset with the extracted embeddings and the same features as the perturbed dataset, plus new features for the embeddings.
    preprocessed_dataset: DatasetDict = perturbed_datasets.map(
        embedding_extraction_fn,
        with_indices=True,
        remove_columns=[dataset_config.audio_column, perturbed_dataset_config.perturbed_audio_column],
        features=preprocessed_features,
        desc="Extracting embeddings from perturbed dataset"
    )

    # Upload the preprocessed dataset with embeddings to Hugging Face Hub
    preprocessed_dataset.push_to_hub(preprocessed_dataset_config.path)
    print(f"Preprocessed dataset created and pushed to Hugging Face Hub with name: {preprocessed_dataset_config.path}")


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