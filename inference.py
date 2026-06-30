# Script for performing zero-shot voice conversion (real-time/offline) with Gradio GUI
import os, torch, torchaudio

import gradio as gr
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
device = inference_config.device

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
# -------------------------------------------------

def convert_voice(audio_A_path, audio_B_path):
    if not audio_A_path or not audio_B_path:
        return None

    # Đọc và resample Audio A, B về 24kHz
    waveform_A, sr_A = torchaudio.load(audio_A_path)
    A_24 = F_audio.resample(waveform_A, orig_freq=sr_A, new_freq=24000).to(device)

    waveform_B, sr_B = torchaudio.load(audio_B_path)
    B_24 = F_audio.resample(waveform_B, orig_freq=sr_B, new_freq=24000).to(device)

    # Bắt đầu quá trình suy luận
    with torch.inference_mode(), autocast(device_type=device, dtype=inference_config.amp, enabled=inference_config.amp != torch.float32):
        # A_24 và B_24 hiện có shape (1, T) -> Thêm chiều N (unsqueeze(0)) để thành (1, 1, T)
        source_tensor = B_24.unsqueeze(0)
        reference_tensor = A_24.unsqueeze(0)

        # Forward pass
        converted_waveform = vc_model(source=source_tensor, reference=reference_tensor)

        # Đưa tensor về CPU, chuyển sang numpy array, và trả về với dạng (sample_rate, numpy_array)
        # Gradio Audio component yêu cầu mảng phải ở dạng 1D (âm thanh mono)
        out_wav = converted_waveform.detach().float().squeeze().cpu().numpy()

        # Trả về numpy array trực tiếp cho Gradio xử lý mà không cần lưu ra file mới
        return (24000, out_wav)

# --- GRADIO INTERFACE ---
with gr.Blocks(title="Zero-Shot Voice Conversion") as demo:
    gr.Markdown("## VZS-VC: Zero-Shot Voice Conversion")
    gr.Markdown("Upload a **Reference Audio** (Target Voice) and a **Source Audio** (Speech to convert).")

    with gr.Row():
        with gr.Column():
            ref_audio = gr.Audio(type="filepath", label="Audio A (Reference / Target Voice)")
            src_audio = gr.Audio(type="filepath", label="Audio B (Source / Voice to Convert)")
            convert_btn = gr.Button("Convert Voice", variant="primary")

        with gr.Column():
            output_audio = gr.Audio(label="Converted Audio", type="numpy", interactive=False)

    convert_btn.click(
        fn=convert_voice,
        inputs=[ref_audio, src_audio],
        outputs=[output_audio]
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)