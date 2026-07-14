import os
import sys
import time
import json
import traceback
import argparse
import multiprocessing
from multiprocessing import cpu_count

import torch
import torch.nn.functional as F
import torchaudio
import torchaudio.functional as F_audio
from torch.amp import autocast

import numpy as np
import librosa
import sounddevice as sd
import FreeSimpleGUI as sg

from modules import (
    BigVGAN,
    EncoderModel,
    ERes2NetV2,
    FCPE,
    LocalRMSAmplitude,
    VoiceConversionModel,
    VoiceGenerator
)
from utils.configs import InferenceConfig, VoiceGeneratorModuleConfig
from utils.logger import load_checkpoint
from utils.modules import (
    load_amplitude_encoder,
    load_content_encoder,
    load_generator,
    load_pitch_encoder,
    load_timbre_encoder,
    load_vocoder
)

os.environ["OMP_NUM_THREADS"] = "4"
if sys.platform == "darwin":
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

device = None
inference_config = None
flag_vc = False

def load_models(args):
    global device, inference_config
    print("Loading VZS-VC Models... Please wait.")
    checkpoint_path = args.checkpoint_path if args.checkpoint_path else "checkpoints/generator.pth"
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
    return vc_model

@torch.inference_mode()
def custom_infer(model, reference_wav, input_wav):
    ref_tensor = torch.from_numpy(reference_wav).float().to(device).view(1, 1, -1)
    src_tensor = input_wav.float().to(device).view(1, 1, -1)
    
    with autocast(device_type=device, dtype=inference_config.amp, enabled=inference_config.amp != torch.float32):
        converted_waveform = model(source=src_tensor, reference=ref_tensor)
    
    return converted_waveform.squeeze()

class Config:
    def __init__(self):
        self.device = device

class GUIConfig:
    def __init__(self) -> None:
        self.reference_audio_path: str = ""
        self.block_time: float = 0.25 
        self.crossfade_time: float = 0.05
        self.I_noise_reduce: bool = False
        self.O_noise_reduce: bool = False
        self.sg_hostapi: str = ""
        self.wasapi_exclusive: bool = False
        self.sg_input_device: str = ""
        self.sg_output_device: str = ""
        self.samplerate: int = 24000
        self.channels: int = 1

class GUI:
    def __init__(self, args) -> None:
        self.gui_config = GUIConfig()
        self.config = Config()
        self.function = "vc"
        self.delay_time = 0
        self.hostapis = None
        self.input_devices = None
        self.output_devices = None
        self.input_devices_indices = None
        self.output_devices_indices = None
        self.stream = None
        self.model_set = load_models(args)
        self.update_devices()
        self.launcher()

    def load(self):
        try:
            os.makedirs("configs/inuse", exist_ok=True)
            if not os.path.exists("configs/inuse/config.json"):
                with open("configs/inuse/config.json", "w") as j:
                    json.dump({}, j)
            with open("configs/inuse/config.json", "r") as j:
                data = json.load(j)
                if data.get("sg_hostapi") in self.hostapis:
                    self.update_devices(hostapi_name=data["sg_hostapi"])
                    if (
                        data.get("sg_input_device") not in self.input_devices
                        or data.get("sg_output_device") not in self.output_devices
                    ):
                        self.update_devices()
                        data["sg_hostapi"] = self.hostapis[0]
                        data["sg_input_device"] = self.input_devices[
                            self.input_devices_indices.index(sd.default.device[0])
                        ]
                        data["sg_output_device"] = self.output_devices[
                            self.output_devices_indices.index(sd.default.device[1])
                        ]
                else:
                    data["sg_hostapi"] = self.hostapis[0]
                    data["sg_input_device"] = self.input_devices[
                        self.input_devices_indices.index(sd.default.device[0])
                    ]
                    data["sg_output_device"] = self.output_devices[
                        self.output_devices_indices.index(sd.default.device[1])
                    ]
        except:
            with open("configs/inuse/config.json", "w") as j:
                data = {
                    "sg_hostapi": self.hostapis[0],
                    "sg_wasapi_exclusive": False,
                    "sg_input_device": self.input_devices[
                        self.input_devices_indices.index(sd.default.device[0])
                    ],
                    "sg_output_device": self.output_devices[
                        self.output_devices_indices.index(sd.default.device[1])
                    ],
                    "block_time": 0.25,
                    "crossfade_length": 0.05,
                }
        return data

    def launcher(self):
        self.config = Config()
        data = self.load()
        sg.theme("LightBlue3")
        layout = [
            [
                sg.Frame(
                    title="Load reference audio",
                    layout=[
                        [
                            sg.Input(
                                default_text=data.get("reference_audio_path", ""),
                                key="reference_audio_path",
                            ),
                            sg.FileBrowse(
                                "choose an audio file",
                                initial_folder=os.path.join(
                                    os.getcwd(), "examples/reference"
                                ),
                                file_types=[
                                    ("WAV Files", "*.wav"),
                                    ("MP3 Files", "*.mp3"),
                                    ("FLAC Files", "*.flac"),
                                    ("M4A Files", "*.m4a"),
                                    ("OGG Files", "*.ogg"),
                                    ("Opus Files", "*.opus"),
                                ],
                            ),
                        ],
                    ],
                )
            ],
            [
                sg.Frame(
                    layout=[
                        [
                            sg.Text("Device type"),
                            sg.Combo(
                                self.hostapis,
                                key="sg_hostapi",
                                default_value=data.get("sg_hostapi", ""),
                                enable_events=True,
                                size=(20, 1),
                            ),
                            sg.Checkbox(
                                "WASAPI Exclusive Device",
                                key="sg_wasapi_exclusive",
                                default=data.get("sg_wasapi_exclusive", False),
                                enable_events=True,
                            ),
                        ],
                        [
                            sg.Text("Input Device"),
                            sg.Combo(
                                self.input_devices,
                                key="sg_input_device",
                                default_value=data.get("sg_input_device", ""),
                                enable_events=True,
                                size=(45, 1),
                            ),
                        ],
                        [
                            sg.Text("Output Device"),
                            sg.Combo(
                                self.output_devices,
                                key="sg_output_device",
                                default_value=data.get("sg_output_device", ""),
                                enable_events=True,
                                size=(45, 1),
                            ),
                        ],
                        [
                            sg.Button("Reload devices", key="reload_devices"),
                            sg.Text("Sampling rate: 24000 (Locked by VZS-VC)"),
                        ],
                    ],
                    title="Sound Device",
                )
            ],
            [
                sg.Frame(
                    layout=[
                        [
                            sg.Text("Block time"),
                            sg.Slider(
                                range=(0.04, 3.0),
                                key="block_time",
                                resolution=0.02,
                                orientation="h",
                                default_value=data.get("block_time", 0.25),
                                enable_events=True,
                            ),
                        ],
                        [
                            sg.Text("Crossfade length"),
                            sg.Slider(
                                range=(0.02, 0.5),
                                key="crossfade_length",
                                resolution=0.02,
                                orientation="h",
                                default_value=data.get("crossfade_length", 0.05),
                                enable_events=True,
                            ),
                        ]
                    ],
                    title="Performance settings",
                ),
            ],
            [
                sg.Button("Start Voice Conversion", key="start_vc"),
                sg.Button("Stop Voice Conversion", key="stop_vc"),
                sg.Radio(
                    "Input listening",
                    "function",
                    key="im",
                    default=False,
                    enable_events=True,
                ),
                sg.Radio(
                    "Voice Conversion",
                    "function",
                    key="vc",
                    default=True,
                    enable_events=True,
                ),
                sg.Text("Algorithm delay (ms):"),
                sg.Text("0", key="delay_time"),
                sg.Text("Inference time (ms):"),
                sg.Text("0", key="infer_time"),
            ],
        ]
        self.window = sg.Window("VZS-VC Real-Time GUI", layout=layout, finalize=True)
        self.event_handler()

    def event_handler(self):
        global flag_vc
        while True:
            event, values = self.window.read()
            if event == sg.WINDOW_CLOSED:
                self.stop_stream()
                exit()
            if event == "reload_devices" or event == "sg_hostapi":
                self.gui_config.sg_hostapi = values["sg_hostapi"]
                self.update_devices(hostapi_name=values["sg_hostapi"])
                if self.gui_config.sg_hostapi not in self.hostapis:
                    self.gui_config.sg_hostapi = self.hostapis[0]
                self.window["sg_hostapi"].Update(values=self.hostapis)
                self.window["sg_hostapi"].Update(value=self.gui_config.sg_hostapi)
                if (
                    self.gui_config.sg_input_device not in self.input_devices
                    and len(self.input_devices) > 0
                ):
                    self.gui_config.sg_input_device = self.input_devices[0]
                self.window["sg_input_device"].Update(values=self.input_devices)
                self.window["sg_input_device"].Update(
                    value=self.gui_config.sg_input_device
                )
                if self.gui_config.sg_output_device not in self.output_devices:
                    self.gui_config.sg_output_device = self.output_devices[0]
                self.window["sg_output_device"].Update(values=self.output_devices)
                self.window["sg_output_device"].Update(
                    value=self.gui_config.sg_output_device
                )
            if event == "start_vc" and not flag_vc:
                if self.set_values(values) == True:
                    print("CUDA is available:", torch.cuda.is_available())
                    self.start_vc()
                    settings = {
                        "reference_audio_path": values["reference_audio_path"],
                        "sg_hostapi": values["sg_hostapi"],
                        "sg_wasapi_exclusive": values["sg_wasapi_exclusive"],
                        "sg_input_device": values["sg_input_device"],
                        "sg_output_device": values["sg_output_device"],
                        "block_time": values["block_time"],
                        "crossfade_length": values["crossfade_length"],
                    }
                    with open("configs/inuse/config.json", "w") as j:
                        json.dump(settings, j)
                    if self.stream is not None:
                        # Đã loại bỏ extra_time_right giúp delay_time hiển thị chính xác và thấp hơn
                        self.delay_time = (
                            self.stream.latency[-1]
                            + values["block_time"]
                            + values["crossfade_length"]
                            + 0.01
                        )
                    self.window["delay_time"].update(
                        int(np.round(self.delay_time * 1000))
                    )
            elif event in ["vc", "im"]:
                self.function = event
            elif event == "stop_vc" or event != "start_vc":
                self.stop_stream()

    def set_values(self, values):
        if len(values["reference_audio_path"].strip()) == 0:
            sg.popup("Choose an audio file")
            return False
        
        self.set_devices(values["sg_input_device"], values["sg_output_device"])
        self.gui_config.sg_hostapi = values["sg_hostapi"]
        self.gui_config.sg_wasapi_exclusive = values["sg_wasapi_exclusive"]
        self.gui_config.sg_input_device = values["sg_input_device"]
        self.gui_config.sg_output_device = values["sg_output_device"]
        self.gui_config.reference_audio_path = values["reference_audio_path"]
        self.gui_config.block_time = values["block_time"]
        self.gui_config.crossfade_time = values["crossfade_length"]
        return True

    def start_vc(self):
        if device == "mps":
            torch.mps.empty_cache()
        elif device == "cuda":
            torch.cuda.empty_cache()
            
        self.reference_wav, _ = librosa.load(self.gui_config.reference_audio_path, sr=24000)
        self.gui_config.samplerate = 24000
        self.gui_config.channels = self.get_device_channels()
        
        self.zc = self.gui_config.samplerate // 50 
        self.block_frame = (
            int(np.round(self.gui_config.block_time * self.gui_config.samplerate / self.zc)) * self.zc
        )
        self.crossfade_frame = (
            int(np.round(self.gui_config.crossfade_time * self.gui_config.samplerate / self.zc)) * self.zc
        )
        self.sola_buffer_frame = min(self.crossfade_frame, 4 * self.zc)
        self.sola_search_frame = self.zc
        
        # Đã loại bỏ extra_frame (chỉ dùng cho Seed-VC), giúp tối ưu RAM và Tốc độ xử lý
        self.input_wav: torch.Tensor = torch.zeros(
            self.crossfade_frame + self.sola_search_frame + self.block_frame,
            device=self.config.device,
            dtype=torch.float32,
        )
        self.input_wav_denoise: torch.Tensor = self.input_wav.clone()
        
        self.rms_buffer: np.ndarray = np.zeros(4 * self.zc, dtype="float32")
        self.sola_buffer: torch.Tensor = torch.zeros(
            self.sola_buffer_frame, device=self.config.device, dtype=torch.float32
        )
        self.output_buffer: torch.Tensor = self.input_wav.clone()
        
        self.fade_in_window: torch.Tensor = (
            torch.sin(
                0.5 * np.pi * torch.linspace(
                    0.0, 1.0, steps=self.sola_buffer_frame, device=self.config.device, dtype=torch.float32
                )
            ) ** 2
        )
        self.fade_out_window: torch.Tensor = 1 - self.fade_in_window
        
        self.start_stream()

    def start_stream(self):
        global flag_vc
        if not flag_vc:
            flag_vc = True
            if "WASAPI" in self.gui_config.sg_hostapi and self.gui_config.sg_wasapi_exclusive:
                extra_settings = sd.WasapiSettings(exclusive=True)
            else:
                extra_settings = None
            self.stream = sd.Stream(
                callback=self.audio_callback,
                blocksize=self.block_frame,
                samplerate=self.gui_config.samplerate,
                channels=self.gui_config.channels,
                dtype="float32",
                extra_settings=extra_settings,
            )
            self.stream.start()

    def stop_stream(self):
        global flag_vc
        if flag_vc:
            flag_vc = False
            if self.stream is not None:
                self.stream.abort()
                self.stream.close()
                self.stream = None

    @torch.inference_mode()
    def audio_callback(self, indata: np.ndarray, outdata: np.ndarray, frames, times, status):
        global flag_vc
        start_time = time.perf_counter()
        indata = librosa.to_mono(indata.T)

        self.input_wav[:-self.block_frame] = self.input_wav[self.block_frame:].clone()
        self.input_wav[-indata.shape[0]:] = torch.from_numpy(indata).to(self.config.device)

        if self.function == "vc":
            if device == "mps":
                start_event = torch.mps.event.Event(enable_timing=True)
                end_event = torch.mps.event.Event(enable_timing=True)
                torch.mps.synchronize()
            else:
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                torch.cuda.synchronize()
            start_event.record()
            
            infer_wav = custom_infer(
                model=self.model_set,
                reference_wav=self.reference_wav,
                input_wav=self.input_wav 
            )
            
            end_event.record()
            if device == "mps":
                torch.mps.synchronize() 
            else:
                torch.cuda.synchronize() 
                
            elapsed_time_ms = start_event.elapsed_time(end_event)
            print(f"Time taken for VC: {elapsed_time_ms}ms")
            
        elif self.gui_config.I_noise_reduce:
            infer_wav = self.input_wav_denoise.clone()
        else:
            infer_wav = self.input_wav.clone()

        conv_input = infer_wav[None, None, : self.sola_buffer_frame + self.sola_search_frame]

        cor_nom = F.conv1d(conv_input, self.sola_buffer[None, None, :])
        cor_den = torch.sqrt(
            F.conv1d(
                conv_input**2,
                torch.ones(1, 1, self.sola_buffer_frame, device=self.config.device),
            )
            + 1e-8
        )

        tensor = cor_nom[0, 0] / cor_den[0, 0]
        if tensor.numel() > 1:
            if sys.platform == "darwin":
                _, sola_offset = torch.max(tensor, dim=0)
                sola_offset = sola_offset.item()
            else:
                sola_offset = torch.argmax(tensor, dim=0).item()
        else:
            sola_offset = tensor.item()

        infer_wav = infer_wav[sola_offset:]
        infer_wav[: self.sola_buffer_frame] *= self.fade_in_window
        infer_wav[: self.sola_buffer_frame] += self.sola_buffer * self.fade_out_window
        
        self.sola_buffer[:] = infer_wav[self.block_frame : self.block_frame + self.sola_buffer_frame]
        
        outdata[:] = (
            infer_wav[: self.block_frame]
            .repeat(self.gui_config.channels, 1)
            .t()
            .cpu()
            .numpy()
        )

        total_time = time.perf_counter() - start_time
        if flag_vc:
            self.window["infer_time"].update(int(total_time * 1000))

    def update_devices(self, hostapi_name=None):
        global flag_vc
        flag_vc = False
        sd._terminate()
        sd._initialize()
        devices = sd.query_devices()
        hostapis = sd.query_hostapis()
        for hostapi in hostapis:
            for device_idx in hostapi["devices"]:
                devices[device_idx]["hostapi_name"] = hostapi["name"]
        self.hostapis = [hostapi["name"] for hostapi in hostapis]
        if hostapi_name not in self.hostapis:
            hostapi_name = self.hostapis[0]
        self.input_devices = [
            d["name"] for d in devices if d["max_input_channels"] > 0 and d["hostapi_name"] == hostapi_name
        ]
        self.output_devices = [
            d["name"] for d in devices if d["max_output_channels"] > 0 and d["hostapi_name"] == hostapi_name
        ]
        self.input_devices_indices = [
            d["index"] if "index" in d else d["name"] for d in devices if d["max_input_channels"] > 0 and d["hostapi_name"] == hostapi_name
        ]
        self.output_devices_indices = [
            d["index"] if "index" in d else d["name"] for d in devices if d["max_output_channels"] > 0 and d["hostapi_name"] == hostapi_name
        ]

    def set_devices(self, input_device, output_device):
        sd.default.device[0] = self.input_devices_indices[self.input_devices.index(input_device)]
        sd.default.device[1] = self.output_devices_indices[self.output_devices.index(output_device)]
        print(f"Input device: {sd.default.device[0]}:{input_device}")
        print(f"Output device: {sd.default.device[1]}:{output_device}")

    def get_device_channels(self):
        max_input_channels = sd.query_devices(device=sd.default.device[0])["max_input_channels"]
        max_output_channels = sd.query_devices(device=sd.default.device[1])["max_output_channels"]
        return min(max_input_channels, max_output_channels, 2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-path", type=str, default=None, help="Path to the model checkpoint")
    args = parser.parse_args()
    gui = GUI(args)