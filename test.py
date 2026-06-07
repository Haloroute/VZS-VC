# Script for testing zero-shot voice conversion with partial masking
import librosa, os, torch, torchaudio

import torch.nn.functional as F
import torchaudio.functional as F_audio

from torch import Tensor
from torch.amp import autocast

from modules import (
    BigVGAN,
    EncoderModel,
    FCPE,
    LocalRMSAmplitude,
    VoiceGenerator
)
from utils.configs import (
    InferenceConfig,
    VoiceGeneratorModuleConfig
)
from utils.logger import load_checkpoint
from utils.modules import (
    load_amplitude_encoder,
    load_content_encoder,
    load_generator,
    load_pitch_encoder,
    load_vocoder
)

def process_mel(audio: Tensor, model_config: VoiceGeneratorModuleConfig) -> Tensor:
    device = audio.device
    window = torch.hann_window(model_config.n_fft).to(device)
    mel_basis = librosa.filters.mel(
        sr=24000,
        n_fft=model_config.n_fft,
        n_mels=model_config.n_mel_bins
    )
    mel_basis_tensor = torch.from_numpy(mel_basis).float().to(device)

    L = audio.shape[0]
    valid_length = (L // (model_config.hop_length * 2)) * (model_config.hop_length * 2)
    audio_clean = audio[:valid_length]

    pad_amount = model_config.n_fft - model_config.hop_length
    left_pad = pad_amount // 2
    right_pad = pad_amount - left_pad
    audio_padded = F.pad(audio_clean, (left_pad, right_pad), mode='constant', value=0.0)

    stft_complex = torch.stft(
        audio_padded,
        n_fft=model_config.n_fft,
        hop_length=model_config.hop_length,
        win_length=model_config.n_fft,
        window=window,
        center=False,
        normalized=False,
        return_complex=True
    )

    magnitudes = torch.abs(stft_complex)
    mel = torch.matmul(mel_basis_tensor, magnitudes)
    mel = torch.log(torch.clamp(mel, min=1e-5))

    mel = mel.transpose(0, 1)
    return mel

def pad_audio_arrays(audio: Tensor, sr: int, target_mod: int, add_extra_silence: bool) -> Tensor:
    L = audio.shape[-1]
    rem = L % target_mod
    pad_len = (target_mod - rem) if rem != 0 else 0
    
    if add_extra_silence:
        pad_len += int(0.5 * sr)
        
    return F.pad(audio, (0, pad_len), mode='constant', value=0.0)

def test_inference():
    checkpoint_path = "checkpoints/generator.pth"
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found at path: {checkpoint_path}")

    model_config = VoiceGeneratorModuleConfig()
    inference_config = InferenceConfig()
    device = inference_config.device

    content_encoder: EncoderModel = load_content_encoder(device)
    pitch_encoder: FCPE = load_pitch_encoder(device)
    amplitude_encoder: LocalRMSAmplitude = load_amplitude_encoder(device)
    model: VoiceGenerator = load_generator(device)
    vocoder: BigVGAN = load_vocoder(device)

    load_checkpoint(
        checkpoint_path, model, None, None,
        None, None, None, None, None, None
    )

    print("All modules loaded successfully.")

    audio_A_path = input("Enter the path to Audio A: ")
    if not os.path.isfile(audio_A_path):
        raise FileNotFoundError(f"Audio A file not found at path: {audio_A_path}")

    ratio_str = input("Enter mask ratio at the end of the file (0.0 to 1.0): ")
    try:
        mask_ratio = float(ratio_str)
        if not (0.0 <= mask_ratio <= 1.0):
            raise ValueError
    except ValueError:
        raise ValueError("Mask ratio must be a float between 0.0 and 1.0")

    waveform_A, sr_A = torchaudio.load(audio_A_path)
    A_16 = F_audio.resample(waveform_A, orig_freq=sr_A, new_freq=16000).to(device)
    A_24 = F_audio.resample(waveform_A, orig_freq=sr_A, new_freq=24000).to(device)

    # Padding
    A_16 = pad_audio_arrays(A_16, sr=16000, target_mod=320, add_extra_silence=False)
    A_24 = pad_audio_arrays(A_24, sr=24000, target_mod=480, add_extra_silence=False)

    # Calculate masking lengths based on 50Hz frames (T)
    # A_24 is padded to be divisible by 480
    T_A = A_24.shape[-1] // 480
    mask_frames = int(T_A * mask_ratio)
    mask_samples = mask_frames * 480

    # Replace the end of A_24 with absolute silence
    if mask_samples > 0:
        A_24[..., -mask_samples:] = 0.0

    with torch.inference_mode(), autocast(device_type=device, dtype=inference_config.amp, enabled=inference_config.amp != torch.float32):
        
        # 1. Biến đổi Mel-spectrogram cho A_24
        mel_A = process_mel(A_24.squeeze(0), model_config).unsqueeze(0) # (1, 2T_A, n_mel_bins)

        # 2. Extract features qua Encoders cho A_16
        content_A = content_encoder.inference(A_16.unsqueeze(0)) # (1, T'_A, D_content)
        pitch_A = pitch_encoder.inference(A_16.unsqueeze(0)) # (1, T_A)
        amp_A = amplitude_encoder.inference(A_16.unsqueeze(0)) # (1, T_A)

        # 3. Tạo Mask (True tương ứng với phần bị che ở cuối file)
        mask_A = torch.zeros(1, T_A, dtype=torch.bool, device=device)
        if mask_frames > 0:
            mask_A[:, -mask_frames:] = True

        content_length = torch.tensor([content_A.shape[1]], device=device)
        token_length = torch.tensor([T_A], device=device)

        # 4. Đưa qua VoiceGenerator
        output_mel: Tensor = model.forward(
            content=content_A,
            pitch=pitch_A,
            amplitude=amp_A,
            mel=mel_A,
            mask_indices=mask_A,
            content_length=content_length,
            token_length=token_length
        ) # Shape: (1, 2T_A, n_mel_bins)

        print(f"Output Mel-spectrogram shape: {output_mel.shape}")

        # 5. Xuất toàn bộ đầu ra bằng Vocoder
        converted_waveform = vocoder.inference(output_mel.transpose(1, 2)) # (1, n_mel_bins, 2T_A) -> (1, 1, L_converted)
        print(f"Converted audio waveform shape: {converted_waveform.shape}")

        # 6. Export ra file WAV
        output_audio_path = "examples/test_output.wav"
        torchaudio.save(output_audio_path, converted_waveform.detach().squeeze(0).cpu(), sample_rate=24000)
        print(f"Test audio saved to {output_audio_path} with sampling rate {24000}.")

def main():
    print("Running mask test inference...")
    test_inference()

if __name__ == "__main__":
    main()