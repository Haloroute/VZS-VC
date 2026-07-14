# Full-stack model for offline/real-time voice conversion
import librosa, torch

import torch.nn as nn
import torch.nn.functional as F
import torchaudio.functional as F_audio

from torch import Tensor


# Container model that encapsulates all sub-modules (Encoders, Generator, Vocoder,...) and implements the full voice conversion pipeline
class VoiceConversionModel(nn.Module):
    def __init__(
        self,
        content_encoder: nn.Module,
        pitch_encoder: nn.Module,
        amplitude_encoder: nn.Module,
        timbre_encoder: nn.Module,
        generator: nn.Module,
        vocoder: nn.Module,
        model_config
    ):
        super().__init__()
        self.content_encoder = content_encoder
        self.pitch_encoder = pitch_encoder
        self.amplitude_encoder = amplitude_encoder
        self.timbre_encoder = timbre_encoder
        self.generator = generator
        self.vocoder = vocoder
        self.model_config = model_config

        # Chuẩn bị mel_basis tensor dưới dạng buffer để tự động chuyển theo thiết bị (device)
        mel_basis = librosa.filters.mel(
            sr=24000,
            n_fft=model_config.n_fft,
            n_mels=model_config.n_mel_bins
        )
        self.register_buffer('mel_basis', torch.from_numpy(mel_basis).float())
        
        # Cửa sổ (window) cũng được đăng ký làm buffer
        window = torch.hann_window(model_config.n_fft)
        self.register_buffer('window', window)

    def process_mel_batch(self, audio: Tensor) -> Tensor:
        """
        Trích xuất Mel-spectrogram hỗ trợ batch processing
        Input: audio tensor shape (N, L)
        Output: mel tensor shape (N, 2T, n_mel_bins)
        """
        # 1. Truncation
        L = audio.shape[-1]
        valid_length = (L // (self.model_config.hop_length * 2)) * (self.model_config.hop_length * 2)
        audio_clean = audio[:, :valid_length]

        # 2. Padding
        pad_amount = self.model_config.n_fft - self.model_config.hop_length
        left_pad = pad_amount // 2
        right_pad = pad_amount - left_pad
        audio_padded = F.pad(audio_clean, (left_pad, right_pad), mode='constant', value=0.0)

        # 3. Native STFT Extraction
        stft_complex = torch.stft(
            audio_padded,
            n_fft=self.model_config.n_fft,
            hop_length=self.model_config.hop_length,
            win_length=self.model_config.n_fft,
            window=self.window,
            center=False,
            normalized=False,
            return_complex=True
        )

        # 4. Magnitude to Mel
        magnitudes = torch.abs(stft_complex)
        mel = torch.matmul(self.mel_basis, magnitudes)
        mel = torch.log(torch.clamp(mel, min=1e-5))

        # Đưa về shape (N, 2T, n_mel_bins)
        mel = mel.transpose(1, 2)
        return mel

    def pad_audio_batch(self, audio: Tensor, sr: int, target_mod: int, add_extra_silence: bool) -> Tensor:
        """
        Bổ sung khoảng lặng (padding) để số khung hình chia hết cho target_mod.
        Input: (N, 1, L)
        """
        L = audio.shape[-1]
        rem = L % target_mod
        pad_len = (target_mod - rem) if rem != 0 else 0

        if add_extra_silence:
            pad_len += int(0.5 * sr)

        return F.pad(audio, (0, pad_len), mode='constant', value=0.0)

    def forward(self, source: Tensor, reference: Tensor) -> Tensor:
        """
        Thực hiện zero-shot voice conversion cho batch dữ liệu.
        Args:
            source: Waveform tensor (N, 1, T_src) ở định dạng 24kHz (Giọng nói cần chuyển đổi - Audio B)
            reference: Waveform tensor (N, 1, T_ref) ở định dạng 24kHz (Giọng nói mục tiêu - Audio A)
        Returns:
            converted_waveform: Waveform tensor (N, 1, T_converted) ở định dạng 24kHz
        """
        device = source.device
        sr_in = 24000
        N = source.shape[0]
        
        # 1. Resample về 16kHz cho các Encoders
        ref_16 = F_audio.resample(reference, orig_freq=sr_in, new_freq=16000)
        src_16 = F_audio.resample(source, orig_freq=sr_in, new_freq=16000)

        # 2. Padding các mảng âm thanh
        # Reference (A) bổ sung chia hết cho 320, 480 và thêm 0.5s khoảng lặng
        ref_16 = self.pad_audio_batch(ref_16, sr=16000, target_mod=320, add_extra_silence=True)
        ref_24 = self.pad_audio_batch(reference, sr=24000, target_mod=480, add_extra_silence=True)

        # Source (B) bổ sung chia hết cho 320, 480 (không thêm 0.5s)
        src_16 = self.pad_audio_batch(src_16, sr=16000, target_mod=320, add_extra_silence=False)
        src_24 = self.pad_audio_batch(source, sr=24000, target_mod=480, add_extra_silence=False)

        # Thay thế src_24 bằng khoảng lặng hoàn toàn (để ép mô hình sinh lại âm thanh này)
        src_24_silence = torch.zeros_like(src_24)

        # 3. Trích xuất Mel-spectrogram cho phần điều kiện
        mel_ref = self.process_mel_batch(ref_24.squeeze(1)) # (N, 2T_A, n_mel_bins)
        mel_src = self.process_mel_batch(src_24_silence.squeeze(1)) # (N, 2T_B, n_mel_bins)

        # 4. Trích xuất các đặc trưng qua Encoders
        content_ref = self.content_encoder.inference(ref_16) # (N, T'_A, D_content)
        content_src = self.content_encoder.inference(src_16) # (N, T'_B, D_content)

        pitch_ref = self.pitch_encoder.inference(ref_16) # (N, T_A)
        pitch_src = self.pitch_encoder.inference(src_16) # (N, T_B)

        amp_ref = self.amplitude_encoder.inference(ref_16) # (N, T_A)
        amp_src = self.amplitude_encoder.inference(src_16) # (N, T_B)

        # Trích xuất âm sắc từ Reference (Target Voice) thay vì Source như trong inference.py
        # để đảm bảo giọng đầu ra mang âm sắc của người mục tiêu
        timbre_ref = self.timbre_encoder.inference(ref_16) # (N, D_timbre)

        # 5. Biến đổi tỉ lệ pitch (Batch-wise processing)
        for i in range(N):
            p_ref_valid = pitch_ref[i][pitch_ref[i] != 0]
            p_src_valid = pitch_src[i][pitch_src[i] != 0]
            
            if len(p_ref_valid) > 0 and len(p_src_valid) > 0:
                p_ref_median = p_ref_valid.median()
                p_src_median = p_src_valid.median()
                pitch_src[i] = pitch_src[i] * (p_ref_median / p_src_median + 1e-5)

        # 6. Ghép nối (Concatenate) các tensor theo chiều thời gian
        content_concat = torch.cat([content_ref, content_src], dim=1) # (N, T'_A + T'_B, D_content)
        pitch_concat = torch.cat([pitch_ref, pitch_src], dim=1) # (N, T_A + T_B)
        amplitude_concat = torch.cat([amp_ref, amp_src], dim=1) # (N, T_A + T_B)
        mel_concat = torch.cat([mel_ref, mel_src], dim=1) # (N, 2T_A + 2T_B, n_mel_bins)

        # 7. Tính toán độ dài T và tạo Mask
        T_ref = pitch_ref.shape[1]
        T_src = pitch_src.shape[1]

        # Mask: False cho Reference (giữ nguyên), True cho Source (che đi để generator sinh lại)
        mask_ref = torch.zeros(N, T_ref, dtype=torch.bool, device=device)
        mask_src = torch.ones(N, T_src, dtype=torch.bool, device=device)
        mask_indices = torch.cat([mask_ref, mask_src], dim=1) # (N, T_A + T_B)

        content_length = torch.full((N,), content_concat.shape[1], device=device, dtype=torch.long)
        token_length = torch.full((N,), T_ref + T_src, device=device, dtype=torch.long)

        # 8. Đưa qua VoiceGenerator để tổng hợp Mel-spectrogram
        output_mel = self.generator(
            content=content_concat,
            pitch=pitch_concat,
            amplitude=amplitude_concat,
            timbre=timbre_ref,

            mel=mel_concat,
            mask_indices=mask_indices,
            content_length=content_length,
            token_length=token_length
        ) # Shape: (N, 2T_A + 2T_B, n_mel_bins)

        # 9. Tách phần Mel-spectrogram ứng với Source đã được biến đổi
        converted_mel_src = output_mel[:, 2 * T_ref:, :] # Shape: (N, 2T_B, n_mel_bins)

        # 10. Chuyển đổi Mel-spectrogram thành dạng sóng 24kHz bằng Vocoder
        converted_waveform = self.vocoder.inference(converted_mel_src.transpose(1, 2)) # (N, n_mel_bins, 2T_B) -> (N, 1, L_converted)
        
        return converted_waveform