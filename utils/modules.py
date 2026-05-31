# Utility functions for loading and saving modules, and other common operations related to the modules.
import torch

from dataclasses import asdict
from safetensors.torch import load_model

from .configs import (
    ERes2NetV2ModuleConfig,
    FCPEModuleConfig,
    LocalRMSModuleConfig,
    NeuCodecModuleConfig,
    VoiceGeneratorModuleConfig,
    Zipformer2ModuleConfig
)
from modules import (
    Conv2dSubsampling,
    EncoderModel,
    ERes2NetV2,
    FCPE,
    LocalRMSAmplitude,
    NeuCodec,
    VoiceGenerator,
    Zipformer2
)


# Functions to load pretrained model for timbre encoder (ERes2Net-V2)
def load_timbre_encoder(device: torch.device, config: ERes2NetV2ModuleConfig = None) -> ERes2NetV2:
    # If no config is provided, use the default one
    if config is None:
        config = ERes2NetV2ModuleConfig()

    # Initialize the Timbre Encoder model
    model = ERes2NetV2(
        n_mel_bins=config.n_mel_bins,
        sampling_rate=config.sampling_rate
    )

    # Load the pretrained checkpoint
    load_model(model, config.checkpoint_path)

    # Set the model to evaluation mode
    model.to(device).eval()
    return model


# Functions to load pretrained model for pitch encoder (FCPE)
def load_pitch_encoder(device: torch.device, config: FCPEModuleConfig = None) -> FCPE:
    # If no config is provided, use the default one
    if config is None:
        config = FCPEModuleConfig()

    # Initialize the Pitch Encoder model
    model = FCPE(**asdict(config))

    # Set the model to evaluation mode
    model.to(device).eval()
    return model


# Functions to load pretrained model for content encoder (Zipformer2)
def load_content_encoder(device: torch.device, config: Zipformer2ModuleConfig = None) -> Zipformer2:
    # If no config is provided, use the default one
    if config is None:
        config = Zipformer2ModuleConfig()

    # 1. Khởi tạo module Subsampling (encoder_embed)
    # Tham số mặc định: feature_dim = 80, encoder_dim đầu tiên = 192
    encoder_embed = Conv2dSubsampling()

    # 2. Khởi tạo mạng chính Zipformer2 (encoder)
    # Các tham số được lấy từ cấu hình mặc định của parser
    encoder = Zipformer2()

    # 3. Khởi tạo AsrModel đóng gói
    # Tắt transducer và attention_decoder, chỉ bật CTC để thoả mãn assert
    # Lấy max của encoder_dim làm chiều đầu ra cho encoder_dim của model
    model = EncoderModel(
        encoder_embed=encoder_embed,
        encoder=encoder,
        dither=config.dither,
        high_freq=config.high_freq,
        n_mel_bins=config.n_mel_bins,
        sampling_rate=config.sampling_rate,
        snip_edges=config.snip_edges
    )

    # Load the pretrained checkpoint
    load_model(model, config.checkpoint_path)

    # Set the model to evaluation mode
    model.to(device).eval()
    return model


# Functions to load module for amplitude encoder (LocalRMS)
def load_amplitude_encoder(device: torch.device, config: LocalRMSModuleConfig = None) -> LocalRMSAmplitude:
    # If no config is provided, use the default one
    if config is None:
        config = LocalRMSModuleConfig()

    model = LocalRMSAmplitude(**asdict(config))

    # Set the model to evaluation mode
    model.to(device).eval()
    return model


# Functions to load module for neural codec (NeuCodec)
def load_codec(device: torch.device, config: NeuCodecModuleConfig = None) -> NeuCodec:
    # If no config is provided, use the default one
    if config is None:
        config = NeuCodecModuleConfig()

    model = NeuCodec.from_pretrained(**asdict(config))

    # Set the model to evaluation mode
    model.to(device).eval()
    return model


# Functions to load module for the main VC model (VoiceGenerator)
def load_generator(device: torch.device, config: VoiceGeneratorModuleConfig = None) -> VoiceGenerator:
    # If no config is provided, use the default one
    if config is None:
        config = VoiceGeneratorModuleConfig()

    model = VoiceGenerator(**asdict(config))

    # Set the model to evaluation mode
    model.to(device).eval()
    return model