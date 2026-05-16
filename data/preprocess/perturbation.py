# Utilities for DSP-based Perturbation Pipeline with F0-Adaptive Shifting and Exclusion Zone
import random, torch

import numpy as np
import pyworld as pw
import torchaudio.functional as F


# Hàm tạo nhiễu loạn cho giọng nói bằng cách phân rã và tổng hợp lại với các tham số được điều chỉnh theo F0 trung bình của đoạn audio
def dsp_perturbate(
    audio_np: np.ndarray, 
    sampling_rate: int = 16000,
    pitch_shift_down_range: tuple = (-6.5, -3.5),
    pitch_shift_up_range: tuple = (-6.5, -3.5),
    formant_shift_down_range: tuple = (1.10, 1.20), # Giả lập người lớn hơn (giọng trầm hơn)
    formant_shift_up_range: tuple = (0.80, 0.90),   # Giả lập người nhỏ hơn (giọng thanh hơn)
    f0_high_thresh: float = 200.0,
    f0_low_thresh: float = 110.0,
    eq_center_freq_range: tuple = (100.0, 8000.0),
    eq_gain_range: tuple = (-12.0, 12.0),
    eq_q_range: tuple = (1.5, 4.0),
    seed: int = None
) -> np.ndarray:
    """
    Tạo nhiễu loạn cho giọng nói bằng cách phân rã và tổng hợp lại với các tham số được điều chỉnh theo F0 trung bình của đoạn audio.
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        
    audio_np = audio_np.astype(np.float64)
    
    # =========================================================================
    # BƯỚC 1: Phân rã
    # =========================================================================
    _f0, t = pw.dio(audio_np, sampling_rate)  # Estimation of F0
    f0 = pw.stonemask(audio_np, _f0, t, sampling_rate)
    sp = pw.cheaptrick(audio_np, f0, t, sampling_rate)
    ap = pw.d4c(audio_np, f0, t, sampling_rate)

    # =========================================================================
    # BƯỚC 2: Nhiễu loạn thích ứng theo F0 (F0-Adaptive Perturbation)
    # =========================================================================
    # Tính F0 trung bình (bỏ qua các đoạn im lặng F0 = 0)
    voiced_f0 = f0[f0 > 0]
    f0_mean = np.mean(voiced_f0) if len(voiced_f0) > 0 else 0

    # Quyết định hướng dịch chuyển
    if f0_mean > f0_high_thresh:
        # Giọng nữ/trẻ em: Chỉ được phép dịch xuống để tránh the thé
        direction = "down"
    elif f0_mean < f0_low_thresh and f0_mean > 0:
        # Giọng nam trầm: Chỉ được phép dịch lên để tránh vỡ tiếng trầm
        direction = "up"
    else:
        # Giọng trung bình: Được phép random 1 trong 2 hướng
        direction = random.choice(["up", "down"])

    # Lấy hệ số theo hướng đã chọn (đảm bảo không bao giờ rơi vào vùng cấm)
    if direction == "down":
        semitones = random.uniform(*pitch_shift_down_range)
        k2 = random.uniform(*formant_shift_down_range)
    else:
        semitones = random.uniform(*pitch_shift_up_range)
        k2 = random.uniform(*formant_shift_up_range)

    # 2a. Dịch Pitch
    k1 = 2.0 ** (semitones / 12.0)
    f0_shifted = f0 * k1
    
    # 2b. Dịch Formant
    num_frames, num_bins = sp.shape
    sp_warped = np.zeros_like(sp)
    freq_bins = np.arange(num_bins)
    warped_bins = freq_bins * k2
    
    for i in range(num_frames):
        sp_warped[i, :] = np.interp(warped_bins, freq_bins, sp[i, :])
        
    # =========================================================================
    # BƯỚC 3 & 4: Tổng hợp và Lọc EQ
    # =========================================================================
    audio_synthesized = pw.synthesize(f0_shifted, sp_warped, ap, sampling_rate, pw.default_frame_period)
    
    audio_tensor = torch.from_numpy(audio_synthesized).float().unsqueeze(0)
    center_freq = random.uniform(*eq_center_freq_range)
    gain = random.uniform(*eq_gain_range)
    q_factor = random.uniform(*eq_q_range)
    
    audio_eq = F.equalizer_biquad(audio_tensor, sampling_rate, center_freq, gain, q_factor)
    
    return audio_eq.squeeze(0).numpy() # (T,)