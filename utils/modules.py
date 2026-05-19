# Utility functions for loading and saving modules, and other common operations related to the modules.
import torch
import torch.nn as nn

from dataclasses import asdict
from safetensors.torch import load_model

from modules import (
    Conv2dSubsampling,
    EncoderModel,
    ERes2NetV2,
    FCPE,
    LocalRMSAmplitude,
    NeuCodec,
    ScheduledFloat,
    Zipformer2
)
from .configs import (
    ERes2NetV2ModuleConfig,
    FCPEModuleConfig,
    LocalRMSModuleConfig,
    NeuCodecModuleConfig,
    Zipformer2ModuleConfig
)


# Functions to load pretrained model for timbre encoder (ERes2Net-V2)
def load_timbre_encoder(config: ERes2NetV2ModuleConfig, device: torch.device):
    # Initialize the Timbre Encoder model
    model = ERes2NetV2()
    
    # Load the pretrained checkpoint
    load_model(model, config.checkpoint_path)

    # Set the model to evaluation mode
    model.to(device).eval()
    return model


# Functions to load pretrained model for pitch encoder (FCPE)
def load_pitch_encoder(config: FCPEModuleConfig, device: torch.device):
    # Initialize the Pitch Encoder model
    model = FCPE(**asdict(config))

    # Set the model to evaluation mode
    model.to(device).eval()
    return model


# Functions to load pretrained model for content encoder (Zipformer2)
def load_content_encoder(config: Zipformer2ModuleConfig, device: torch.device):
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
        encoder=encoder
    )

    # Load the pretrained checkpoint
    load_model(model, config.checkpoint_path)

    # Set the model to evaluation mode
    model.to(device).eval()
    return model


# Functions to load module for amplitude encoder (LocalRMS)
def load_amplitude_encoder(config: LocalRMSModuleConfig, device: torch.device):
    model = LocalRMSAmplitude(**asdict(config))

    # Set the model to evaluation mode
    model.to(device).eval()
    return model


# Functions to load module for neural codec (NeuCodec)
def load_codec(config: NeuCodecModuleConfig, device: torch.device):
    model = NeuCodec.from_pretrained(**asdict(config))

    # Set the model to evaluation mode
    model.to(device).eval()
    return model