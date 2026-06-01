# Script for performing zero-shot voice conversion (real-time/offline)
import os, torch, torchaudio
import torchaudio.functional as F_audio

from torch import Tensor
from torch.amp import autocast

from modules import (
    EncoderModel,
    FCPE,
    LocalRMSAmplitude,
    NeuCodec,
    VoiceGenerator
)
from utils.configs import (
    InferenceConfig,
    VoiceGeneratorModuleConfig
)
from utils.logger import load_checkpoint
from utils.modules import (
    load_amplitude_encoder,
    load_codec,
    load_content_encoder,
    load_generator,
    load_pitch_encoder
)


# Function to perform offline zero-shot voice conversion using a trained model
def inference_offline():
    # Prompt the user to enter the path to the checkpoint file
    checkpoint_path = input("Enter the path to the checkpoint file: ")
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found at path: {checkpoint_path}")

    # Load the configurations
    model_config = VoiceGeneratorModuleConfig()
    inference_config = InferenceConfig()

    # Load the pretrained modules and the main VC model
    content_encoder: EncoderModel = load_content_encoder(inference_config.device)
    pitch_encoder: FCPE = load_pitch_encoder(inference_config.device)
    amplitude_encoder: LocalRMSAmplitude = load_amplitude_encoder(inference_config.device)
    codec: NeuCodec = load_codec(inference_config.device)
    model: VoiceGenerator = load_generator(inference_config.device)
    load_checkpoint(
        checkpoint_path,
        model, None, None, None, None
    ) # We only need to load the model weights for inference, so we can pass None for the EMA model, optimizer, scheduler, and scaler.

    print("All modules loaded successfully.")

    # Load the source audio (audio that needs to be converted) and target audio (audio that needs to be converted to) file for conversion
    source_audio_path = input("Enter the path to the source audio file (the audio that needs to be converted): ")
    target_audio_path = input("Enter the path to the target audio file (the audio that needs to be converted to): ")
    if not os.path.isfile(source_audio_path):
        raise FileNotFoundError(f"Source audio file not found at path: {source_audio_path}")
    if not os.path.isfile(target_audio_path):
        raise FileNotFoundError(f"Target audio file not found at path: {target_audio_path}")

    source_waveform, source_sr = torchaudio.load(source_audio_path) # (1, T_source)
    if source_sr != inference_config.input_sampling_rate:
        source_waveform = F_audio.resample(source_waveform, orig_freq=source_sr, new_freq=inference_config.input_sampling_rate)
    source_waveform = source_waveform.unsqueeze(0).to(inference_config.device) # (1, T_source) -> (1, 1, T_source)
    print(f"Loaded source audio from {source_audio_path} with sampling rate {source_sr} and waveform shape {source_waveform.shape}.")

    target_waveform, target_sr = torchaudio.load(target_audio_path) # (1, T_target)
    if target_sr != inference_config.input_sampling_rate:
        target_waveform = F_audio.resample(target_waveform, orig_freq=target_sr, new_freq=inference_config.input_sampling_rate)
    target_waveform = target_waveform.unsqueeze(0).to(inference_config.device) # (1, T_target) -> (1, 1, T_target)
    print(f"Loaded target audio from {target_audio_path} with sampling rate {target_sr} and waveform shape {target_waveform.shape}.")

    # Begin the inference process
    with torch.inference_mode(), autocast(device_type=inference_config.device, dtype=inference_config.amp, enabled=inference_config.amp != torch.float32):
        # Extract the content, pitch, amplitude from the source audio and the timbre from the target audio using the respective pretrained modules
        timbre_embedding = codec.inference(target_waveform.to(inference_config.device)) # (1, T_timbre, D_timbre)
        print(f"Extracted timbre embedding from target audio with shape {timbre_embedding.shape}.")

        content_embedding = content_encoder.inference(source_waveform.to(inference_config.device)) # (1, T_content, D_content)
        print(f"Extracted content embedding from source audio with shape {content_embedding.shape}.")

        pitch_embedding = pitch_encoder.inference(source_waveform.to(inference_config.device)) # (1, T_pitch)
        print(f"Extracted pitch embedding from source audio with shape {pitch_embedding.shape}.")

        amplitude_embedding = amplitude_encoder.inference(source_waveform.to(inference_config.device)) # (1, T_amplitude)
        print(f"Extracted amplitude embedding from source audio with shape {amplitude_embedding.shape}.")

        # Create shape and length tensors for the content, pitch, amplitude, and timbre embeddings
        content_length = torch.tensor([content_embedding.shape[1]], device=inference_config.device) # (1,) tensor containing the length of the content embedding sequence
        pitch_length = torch.tensor([pitch_embedding.shape[1]], device=inference_config.device) # (1,) tensor containing the length of the pitch embedding sequence
        amplitude_length = torch.tensor([amplitude_embedding.shape[1]], device=inference_config.device) # (1,) tensor containing the length of the amplitude embedding sequence
        timbre_length = torch.tensor([timbre_embedding.shape[1]], device=inference_config.device) # (1,) tensor containing the length of the timbre embedding sequence

        target_length = torch.max(torch.stack([content_embedding, pitch_embedding, amplitude_embedding, timbre_embedding]), dim=0).values - 1
        # (1,) tensor containing the maximum length among the content, pitch, amplitude embedding sequences
        target_shape = torch.Tensor.shape(1, torch.max(target_length).item(), model_config.d_codec) # The shape of the target tensor for the model input, which is (N, T, D_codec). Here N=1 for inference.

        # Perform inference with the loaded model to get the output codec embedding for the converted audio
        output: Tensor = model.forward(
            content=content_embedding,
            pitch=pitch_embedding,
            amplitude=amplitude_embedding,
            timbre=timbre_embedding,

            content_length=content_length,
            pitch_length=pitch_length,
            amplitude_length=amplitude_length,
            timbre_length=timbre_length,

            target_shape=target_shape,
            target_length=target_length
        ) # (1, N_bins, T, D_codec)

    # Post-process the output codec embedding to get the converted audio waveform and save it to a file
    # Perform argmax over the N_bins dimension to get the quantized codec embedding
    quantized_codec_embedding = torch.argmax(output, dim=1) # (1, T, D_codec)

    # Convert the quantized codec embedding to the original audio waveform
    codec_weights = torch.pow(model_config.n_bins, torch.arange(model_config.d_codec)) # (D_codec,)
    codec_weights = codec_weights.to(dtype=torch.long, device=inference_config.device)
    # (D_codec,) tensor containing the weights for each dimension of the codec embedding based on the number of bins

    # Multiply the quantized codec embedding with the weights and sum over the D_codec dimension to get the quantized indices for each time step
    quantized_indices = torch.sum(quantized_codec_embedding * codec_weights, dim=-1).unsqueeze(1) # (1, 1, T) tensor containing the quantized indices for each time step

    # Decode the quantized indices to get the converted audio waveform
    converted_waveform = codec.decode_code(quantized_indices) # (1, 1, T_converted) tensor containing the converted audio waveform
    print(f"Converted audio waveform shape: {converted_waveform.shape}")

    # Export the converted audio waveform to a file
    output_audio_path = input("Enter the path to save the converted audio file (e.g., output.wav): ")
    torchaudio.save(output_audio_path, converted_waveform.squeeze(0).cpu(), sample_rate=inference_config.output_sampling_rate)
    print(f"Converted audio saved to {output_audio_path} with sampling rate {inference_config.output_sampling_rate}.")


# Main function to run the inference script
def main():
    print("This is the main inference script. You can perform zero-shot voice conversion task (real-time/offline) here.")
    print("Choose an action:")
    print("1. Offline zero-shot voice conversion")
    print("2. Real-time zero-shot voice conversion (not implemented yet)")
    # choice = input("Enter the number of your choice: ")
    choice = "1" # For now, we will directly call the offline inference function since the real-time one is not implemented yet.
    if choice == "1":
        print("You chose to perform zero-shot voice conversion.")
        inference_offline()
    elif choice == "2":
        print("You chose to perform real-time zero-shot voice conversion.")
        raise NotImplementedError("Real-time zero-shot voice conversion is not implemented yet.")
    else:
        print("Invalid choice. Please run the script again and choose a valid option.")

if __name__ == "__main__":
    main()