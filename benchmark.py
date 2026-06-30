import os
import time
import random
import glob
import torch
import torchaudio
import torchaudio.functional as F_audio

from torch.amp import autocast

from modules import (
    BigVGAN,
    EncoderModel,
    ERes2NetV2,
    FCPE,
    LocalRMSAmplitude,
    VoiceConversionModel,
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
    load_timbre_encoder,
    load_vocoder
)

# --- GLOBAL SETUP: Load models once at startup ---
print("Loading models... Please wait.")
checkpoint_path = "checkpoints/generator.pth"
if not os.path.isfile(checkpoint_path):
    raise FileNotFoundError(f"Checkpoint file not found at path: {checkpoint_path}")

model_config = VoiceGeneratorModuleConfig()
inference_config = InferenceConfig()
device = 'cpu'

content_encoder: EncoderModel = load_content_encoder(device)
pitch_encoder: FCPE = load_pitch_encoder(device)
amplitude_encoder: LocalRMSAmplitude = load_amplitude_encoder(device)
timbre_encoder: ERes2NetV2 = load_timbre_encoder(device)
generator: VoiceGenerator = load_generator(device)
vocoder: BigVGAN = load_vocoder(device)

load_checkpoint(
    checkpoint_path, None, generator, None,
    None, None, None, None, None, None
)

# Khởi tạo mô hình Voice Conversion
vc_model = VoiceConversionModel(
    content_encoder=content_encoder,
    pitch_encoder=pitch_encoder,
    amplitude_encoder=amplitude_encoder,
    timbre_encoder=timbre_encoder,
    generator=generator,
    vocoder=vocoder,
    model_config=model_config
).to(device)
vc_model.eval()

print("All modules loaded successfully.")
print("-" * 40)
# -------------------------------------------------

def run_benchmark(directory, num_test_cases, output_dir="output_tests"):
    # Tìm tất cả các định dạng audio phổ biến trong thư mục
    audio_extensions = ['*.wav', '*.mp3', '*.flac', '*.m4a']
    audio_files = []
    for ext in audio_extensions:
        audio_files.extend(glob.glob(os.path.join(directory, ext)))
        
    if len(audio_files) < 2:
        print(f"Lỗi: Không tìm thấy đủ file audio trong {directory}. Cần ít nhất 2 file để thực hiện biến đổi.")
        return

    # Tạo thư mục lưu kết quả nếu chưa có
    os.makedirs(output_dir, exist_ok=True)

    total_processing_time = 0.0
    total_audio_duration = 0.0

    print(f"Bắt đầu chạy {num_test_cases} test cases từ thư mục: {directory}")
    print(f"Kết quả âm thanh đầu ra sẽ được lưu tại: {output_dir}/")
    print("-" * 40)

    for i in range(num_test_cases):
        # Lấy ngẫu nhiên 2 file audio
        file_A, file_B = random.sample(audio_files, 2)
        
        # Đọc file A (Reference)
        waveform_A, sr_A = torchaudio.load(file_A)
        A_24 = F_audio.resample(waveform_A, orig_freq=sr_A, new_freq=24000).to(device)
        duration_A = waveform_A.shape[1] / sr_A
        
        # Đọc file B (Source)
        waveform_B, sr_B = torchaudio.load(file_B)
        B_24 = F_audio.resample(waveform_B, orig_freq=sr_B, new_freq=24000).to(device)
        duration_B = waveform_B.shape[1] / sr_B
        
        # Tổng thời lượng của 2 file
        current_audio_duration = duration_A + duration_B
        total_audio_duration += current_audio_duration

        # Chuẩn bị Tensor
        source_tensor = B_24.unsqueeze(0)
        reference_tensor = A_24.unsqueeze(0)

        # Bắt đầu đo thời gian và suy luận
        start_time = time.time()
        with torch.inference_mode(), autocast(device_type=device if isinstance(device, str) else device.type, dtype=inference_config.amp, enabled=inference_config.amp != torch.float32):
            converted_waveform = vc_model(source=source_tensor, reference=reference_tensor)
        
        # Đồng bộ CUDA nếu dùng GPU để đo thời gian chính xác
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            
        end_time = time.time()
        process_time = end_time - start_time
        total_processing_time += process_time
        
        current_rtf = process_time / current_audio_duration

        # Lưu audio đầu ra
        out_tensor = converted_waveform.detach().float().cpu().view(1, -1)
        out_filename = f"test_{i+1}_ref_{os.path.basename(file_A)}_src_{os.path.basename(file_B)}.wav"
        torchaudio.save(os.path.join(output_dir, out_filename), out_tensor, 24000)

        # In thông số từng test case
        print(f"Test case {i+1}/{num_test_cases}:")
        print(f"  - File Ref (A): {os.path.basename(file_A)} | Thời lượng: {duration_A:.2f}s")
        print(f"  - File Src (B): {os.path.basename(file_B)} | Thời lượng: {duration_B:.2f}s")
        print(f"  - Tổng thời lượng 2 file: {current_audio_duration:.2f}s")
        print(f"  - Thời gian xử lý: {process_time:.4f}s")
        print(f"  => RTF (Case {i+1}): {current_rtf:.4f}")
        print("-" * 30)

    # Tính toán chỉ số RTF trung bình
    avg_rtf = total_processing_time / total_audio_duration

    print("KẾT QUẢ TỔNG QUAN:")
    print(f"Số lượng Test case đã chạy : {num_test_cases}")
    print(f"Tổng thời gian xử lý mô hình: {total_processing_time:.4f} giây")
    print(f"Tổng thời lượng audio đã nạp: {total_audio_duration:.4f} giây")
    print(f"RTF Trung Bình (Average RTF): {avg_rtf:.4f}")


if __name__ == "__main__":
    try:
        input_dir = input("Nhập đường dẫn thư mục chứa file âm thanh: ").strip()
        num_cases_str = input("Nhập số lượng test case cần thực hiện: ").strip()
        
        if not os.path.isdir(input_dir):
            print("Đường dẫn thư mục không hợp lệ!")
        else:
            num_cases = int(num_cases_str)
            if num_cases <= 0:
                print("Số lượng test case phải lớn hơn 0.")
            else:
                run_benchmark(input_dir, num_cases)
    except ValueError:
        print("Lỗi: Số lượng test case phải là số nguyên!")
    except KeyboardInterrupt:
        print("\nĐã hủy quá trình test.")