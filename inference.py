# Script for performing zero-shot voice conversion (real-time/offline)
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
    """
    Trích xuất Mel-spectrogram y hệt cách thức trong file dataset.py
    Input: audio tensor (L,)
    Output: mel tensor (2T, n_mel_bins)
    """
    device = audio.device
    window = torch.hann_window(model_config.n_fft).to(device)
    mel_basis = librosa.filters.mel(
        sr=24000,
        n_fft=model_config.n_fft,
        n_mels=model_config.n_mel_bins
    )
    mel_basis_tensor = torch.from_numpy(mel_basis).float().to(device)

    # 1. Truncation
    L = audio.shape[0]
    valid_length = (L // (model_config.hop_length * 2)) * (model_config.hop_length * 2)
    audio_clean = audio[:valid_length]

    # 2. Padding
    pad_amount = model_config.n_fft - model_config.hop_length
    left_pad = pad_amount // 2
    right_pad = pad_amount - left_pad
    audio_padded = F.pad(audio_clean, (left_pad, right_pad), mode='constant', value=0.0)

    # 3. Native STFT Extraction
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

    # 4. Magnitude to Mel
    magnitudes = torch.abs(stft_complex)
    mel = torch.matmul(mel_basis_tensor, magnitudes)
    mel = torch.log(torch.clamp(mel, min=1e-5))

    # Đưa về shape (2T, n_mel_bins)
    mel = mel.transpose(0, 1)
    return mel

def pad_audio_arrays(audio: Tensor, sr: int, target_mod: int, add_extra_silence: bool) -> Tensor:
    """
    Bổ sung khoảng lặng (padding) để số khung hình chia hết cho target_mod (320 hoặc 480).
    Nếu add_extra_silence = True, bổ sung thêm 0.5s khoảng lặng.
    """
    L = audio.shape[-1]
    rem = L % target_mod
    pad_len = (target_mod - rem) if rem != 0 else 0
    
    if add_extra_silence:
        pad_len += int(0.5 * sr)
        
    return F.pad(audio, (0, pad_len), mode='constant', value=0.0)

# Function to perform offline zero-shot voice conversion using a trained model
def inference_offline():
    # Nhập đường dẫn checkpoint
    checkpoint_path = "checkpoints/generator.pth"
    # checkpoint_path = input("Enter the path to the checkpoint file: ")
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found at path: {checkpoint_path}")

    # Load configurations
    model_config = VoiceGeneratorModuleConfig()
    inference_config = InferenceConfig()
    device = inference_config.device

    # Load pretrained modules
    content_encoder: EncoderModel = load_content_encoder(device)
    pitch_encoder: FCPE = load_pitch_encoder(device)
    amplitude_encoder: LocalRMSAmplitude = load_amplitude_encoder(device)
    model: VoiceGenerator = load_generator(device)
    vocoder: BigVGAN = load_vocoder(device) # Load vocoder để xuất âm thanh

    load_checkpoint(
        checkpoint_path, model, None, None,
        None, None, None, None, None, None
    )

    print("All modules loaded successfully.")

    # Nhập 2 file âm thanh A (Source) và B (Target)
    audio_A_path = input("Enter the path to Audio A (Source/Context): ")
    audio_B_path = input("Enter the path to Audio B (Target to convert): ")
    if not os.path.isfile(audio_A_path):
        raise FileNotFoundError(f"Audio A file not found at path: {audio_A_path}")
    if not os.path.isfile(audio_B_path):
        raise FileNotFoundError(f"Audio B file not found at path: {audio_B_path}")

    # Đọc và resample Audio A
    waveform_A, sr_A = torchaudio.load(audio_A_path)
    A_16 = F_audio.resample(waveform_A, orig_freq=sr_A, new_freq=16000).to(device)
    A_24 = F_audio.resample(waveform_A, orig_freq=sr_A, new_freq=24000).to(device)

    # Đọc và resample Audio B
    waveform_B, sr_B = torchaudio.load(audio_B_path)
    B_16 = F_audio.resample(waveform_B, orig_freq=sr_B, new_freq=16000).to(device)
    B_24 = F_audio.resample(waveform_B, orig_freq=sr_B, new_freq=24000).to(device)

    # Padding: Thêm khoảng lặng vào cuối mỗi mảng
    # A_16, A_24 bổ sung chia hết cho 320, 480 và thêm 0.5s
    A_16 = pad_audio_arrays(A_16, sr=16000, target_mod=320, add_extra_silence=True)
    A_24 = pad_audio_arrays(A_24, sr=24000, target_mod=480, add_extra_silence=True)
    
    # B_16, B_24 bổ sung chia hết cho 320, 480 (không thêm 0.5s)
    B_16 = pad_audio_arrays(B_16, sr=16000, target_mod=320, add_extra_silence=False)
    B_24 = pad_audio_arrays(B_24, sr=24000, target_mod=480, add_extra_silence=False)

    # Thay thế B_24 bằng khoảng lặng hoàn toàn
    B_24 = torch.zeros_like(B_24)

    # Bắt đầu quá trình suy luận
    with torch.inference_mode(), autocast(device_type=device, dtype=inference_config.amp, enabled=inference_config.amp != torch.float32):
        # 1. Biến đổi Mel-spectrogram cho A_24 và B_24
        mel_A = process_mel(A_24.squeeze(0), model_config).unsqueeze(0) # (1, 2T_A, n_mel_bins)
        mel_B = process_mel(B_24.squeeze(0), model_config).unsqueeze(0) # (1, 2T_B, n_mel_bins)

        # 2. Extract features qua Encoders cho A_16 và B_16
        content_A = content_encoder.inference(A_16.unsqueeze(0)) # (1, T'_A, D_content)
        content_B = content_encoder.inference(B_16.unsqueeze(0)) # (1, T'_B, D_content)

        pitch_A = pitch_encoder.inference(A_16.unsqueeze(0)) # (1, T_A)
        pitch_B = pitch_encoder.inference(B_16.unsqueeze(0)) # (1, T_B)

        amp_A = amplitude_encoder.inference(A_16.unsqueeze(0)) # (1, T_A)
        amp_B = amplitude_encoder.inference(B_16.unsqueeze(0)) # (1, T_B)

        # 3. Ghép nối (Concatenate) các tensor theo chiều thời gian
        content_concat = torch.cat([content_A, content_B], dim=1) # (1, T'_A + T'_B, D_content)
        pitch_concat = torch.cat([pitch_A, pitch_B], dim=1) # (1, T_A + T_B)
        amplitude_concat = torch.cat([amp_A, amp_B], dim=1) # (1, T_A + T_B)
        mel_concat = torch.cat([mel_A, mel_B], dim=1) # (1, 2T_A + 2T_B, n_mel_bins)

        # 4. Tính toán T và tạo Mask
        T_A = pitch_A.shape[1]
        T_B = pitch_B.shape[1]

        # Mask: False cho A (giữ nguyên), True cho B (che đi để generator sinh lại)
        mask_A = torch.zeros(1, T_A, dtype=torch.bool, device=device)
        mask_B = torch.ones(1, T_B, dtype=torch.bool, device=device)
        mask_indices = torch.cat([mask_A, mask_B], dim=1) # (1, T_A + T_B)

        content_length = torch.tensor([content_concat.shape[1]], device=device)
        token_length = torch.tensor([T_A + T_B], device=device)

        # 5. Đưa qua VoiceGenerator
        output_mel: Tensor = model.forward(
            content=content_concat,
            pitch=pitch_concat,
            amplitude=amplitude_concat,
            mel=mel_concat,
            mask_indices=mask_indices,
            content_length=content_length,
            token_length=token_length
        ) # Shape: (1, 2T_A + 2T_B, n_mel_bins)

        # 6. Tách phần ứng với mask (âm thanh B sau biến đổi)
        # Mel spectrogram có chiều dài thời gian gấp đôi T, do đó phần của B bắt đầu từ 2 * T_A
        converted_mel_B = output_mel[:, 2 * T_A:, :] # Shape: (1, 2T_B, n_mel_bins)
        print(f"Converted Mel-spectrogram shape: {converted_mel_B.shape}")

        # 7. Xuất phần này làm âm thanh đầu ra bằng Vocoder
        # Mel transpose back to (1, n_mel_bins, 2T_B) for vocoder input if needed, depending on Vocoder architecture
        converted_waveform = vocoder.inference(converted_mel_B.transpose(1, 2)) # (1, 2T_B, n_mel_bins) -> (1, n_mel_bins, 2T_B) -> (1, 1, L_converted)
        print(f"Converted audio waveform shape: {converted_waveform.shape}")

        # 8. Export ra file WAV
        output_audio_path = "examples/output.wav"
        # output_audio_path = input("Enter the path to save the converted audio file (e.g., output.wav): ")
        torchaudio.save(output_audio_path, converted_waveform.detach().squeeze(0).cpu(), sample_rate=24000)
        print(f"Converted audio saved to {output_audio_path} with sampling rate {24000}.")


def main():
    print("This is the main inference script. You can perform zero-shot voice conversion task (real-time/offline) here.")
    print("Choose an action:")
    print("1. Offline zero-shot voice conversion")
    print("2. Real-time zero-shot voice conversion (not implemented yet)")
    
    choice = "1"
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