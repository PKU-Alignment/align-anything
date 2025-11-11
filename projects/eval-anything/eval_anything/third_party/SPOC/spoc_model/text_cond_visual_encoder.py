# Copyright 2024 Allen Institute for AI
# ==============================================================================

import math
from dataclasses import dataclass
from typing import List, Literal, Tuple

import torch
import torch.nn as nn
from open_clip.transformer import TextTransformer
from PIL import Image
from transformers import T5EncoderModel

from eval_anything.third_party.SPOC.utils.sensor_constant_utils import is_a_visual_sensor


@dataclass
class Dinov2Config:
    model: str = 'dinov2_vits14'
    output_size: Tuple[int, int, int] = (384, 7, 12)


class Dinov2(nn.Module):
    def __init__(self, cfg: Dinov2Config):
        super().__init__()
        self.cfg = cfg
        self.model = torch.hub.load('facebookresearch/dinov2', cfg.model)
        self.pool = nn.AdaptiveAvgPool2d(cfg.output_size[1:])
        self.eval()

    def forward(self, x):
        # 2. 移除批次维度
        image_tensor = x.squeeze(0)  # 形状: [3, 360, 628]

        # 3. 调整维度顺序
        # 将 (C, H, W) 格式转换为 (H, W, C) 格式，这是图像库（如Pillow, Matplotlib）期望的格式
        image_tensor_hwc = image_tensor.permute(1, 2, 0)  # 形状: [360, 628, 3]

        # 4. 将数值范围从 [0, 1] 转换为 [0, 255] 并改变数据类型
        # 图像文件通常使用 8-bit 无符号整数 (uint8)
        image_tensor_uint8 = (image_tensor_hwc * 255).byte()

        # 5. 将张量移动到 CPU 并转换为 NumPy 数组
        # PIL 库基于 NumPy 操作
        numpy_array = image_tensor_uint8.cpu().numpy()

        # 6. 从 NumPy 数组创建 PIL 图像对象
        pil_image_manual = Image.fromarray(numpy_array)

        # 7. 保存图像
        file_path_manual = 'output_image_manual.png'
        pil_image_manual.save(file_path_manual)
        assert x.shape[-2:] == (224, 384), f'Expected shape is 224x384; got {x.shape}'
        with torch.no_grad():
            x = self.model.forward_features(x[:, :, :, 3:-3])['x_norm_patchtokens']
            B, _, D = x.shape  # Bx432x384
            x = x.permute(0, 2, 1)  # Bx384x432
            x = x.reshape(B, D, 16, 27)
            x = self.pool(x)
            return x


IMAGE_ENCODERS = dict(
    Dinov2Small=(Dinov2, Dinov2Config()),
    Dinov2Base=(Dinov2, Dinov2Config(model='dinov2_vitb14', output_size=(768, 7, 12))),
)


@dataclass
class TransformerConfig:
    num_layers: int = 3
    d_model: int = 512
    nhead: int = 8

    def to_dict(self):
        return {'num_layers': self.num_layers, 'd_model': self.d_model, 'nhead': self.nhead}


TEXT_ENCODER_DIMS = {
    't5-small': 512,
    't5-base': 768,
    't5-large': 1024,
}


def create_text_encoder(encoder_name):
    if 't5' in encoder_name.lower():
        return T5EncoderModel.from_pretrained(encoder_name)
    else:
        raise NotImplementedError('Only SigLIP and T5 text encoders are supported.')


class TextCondVisualEncoderConfig:
    def __init__(
        self,
        image_encoder: str = 'Dinov2Small',
        text_encoder: str = 't5-small',
        fusion_xformer: dict = None,  # 从 json 加载时，这里会是一个字典
        input_sensors: List[str] = None,
        bbox_encoding_type: Literal['positional'] = 'positional',
        **kwargs,  # 虽然这里暂时没用到，但加上是个好习惯
    ):
        self.image_encoder = image_encoder
        self.text_encoder = text_encoder

        # 关键修复：将传入的字典转换为 TransformerConfig 对象
        self.fusion_xformer = (
            TransformerConfig(**fusion_xformer) if fusion_xformer else TransformerConfig()
        )

        self.input_sensors = input_sensors if input_sensors is not None else []
        self.bbox_encoding_type = bbox_encoding_type

    def to_dict(self):
        return {
            'image_encoder': self.image_encoder,
            'text_encoder': self.text_encoder,
            'fusion_xformer': self.fusion_xformer.to_dict() if self.fusion_xformer else None,
            'input_sensors': self.input_sensors,
            'bbox_encoding_type': self.bbox_encoding_type,
        }


class TextCondMultiCameraVisualEncoder(nn.Module):
    def __init__(self, cfg: TextCondVisualEncoderConfig):
        super().__init__()
        self.cfg = cfg
        if cfg.image_encoder == 'dinov2' and cfg.image_encoder not in IMAGE_ENCODERS:
            cfg.image_encoder = 'Dinov2Small'
            print('REAPLACING DINOV2 WITH DINOV2SMALL')

        if cfg.image_encoder in IMAGE_ENCODERS:
            image_encoder_model_cls, image_encoder_cfg = IMAGE_ENCODERS[cfg.image_encoder]
            self.image_encoder = image_encoder_model_cls(image_encoder_cfg)
        else:
            raise NotImplementedError()

        self.visual_compressor = self.create_compressor()
        self.visual_adapter = nn.Sequential(
            nn.Linear(self.cfg.fusion_xformer.d_model, self.cfg.fusion_xformer.d_model),
            nn.LayerNorm(self.cfg.fusion_xformer.d_model),
            nn.ReLU(),
        )

        self.text_encoder = create_text_encoder(cfg.text_encoder)

        self.text_encoder.eval()

        self.text_adapter = nn.Sequential(
            nn.Linear(TEXT_ENCODER_DIMS[cfg.text_encoder], self.cfg.fusion_xformer.d_model),
            nn.LayerNorm(self.cfg.fusion_xformer.d_model),
            nn.ReLU(),
        )
        self.fusion_xformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=cfg.fusion_xformer.d_model, nhead=cfg.fusion_xformer.nhead, batch_first=True
            ),
            num_layers=cfg.fusion_xformer.num_layers,
        )
        self.fusion_token = nn.Parameter(0.1 * torch.rand(cfg.fusion_xformer.d_model))
        self.visual_sensors = [sensor for sensor in cfg.input_sensors if is_a_visual_sensor(sensor)]
        # KE: This is absolutely important! # KE2: Actually not so much anymore lol
        self.visual_sensors = sorted(self.visual_sensors)
        for sensor in self.visual_sensors:
            setattr(
                self,
                f'visual_sensor_token_{sensor}',
                nn.Parameter(0.1 * torch.rand(cfg.fusion_xformer.d_model)),
            )

    def encode_text(self, preproc_text_input):
        with torch.no_grad():
            if isinstance(self.text_encoder, TextTransformer):
                cls_feats, text_feats = self.text_encoder(preproc_text_input)
                text_feats = torch.cat([text_feats, cls_feats.unsqueeze(1)], dim=1)
            else:
                text_feats = self.text_encoder(**preproc_text_input).last_hidden_state

        return self.text_adapter(text_feats)

    def encode_imgs(self, imgs):
        B, T, C, H, W = imgs.shape
        feats = self.visual_compressor(
            self.image_encoder(imgs.reshape(B * T, C, H, W))
        )  # BTxC_xH_xW_
        _, C_, H_, W_ = feats.shape
        feats = feats.reshape(B * T, C_, H_ * W_).permute(0, 2, 1)  # BTxH_W_xC_
        return self.visual_adapter(feats)

    def create_compressor(self):
        return nn.Sequential(
            nn.Conv2d(self.image_encoder.cfg.output_size[0], self.cfg.fusion_xformer.d_model, 1),
            nn.ReLU(),
            nn.Conv2d(self.cfg.fusion_xformer.d_model, self.cfg.fusion_xformer.d_model, 1),
            nn.ReLU(),
        )

    def get_image_text_feats(self, frames, goals, text_feats):
        all_img_features = {}
        images_chw = None
        for sensor in frames.keys():
            assert is_a_visual_sensor(sensor)
            imgs = frames[sensor]
            B, T, C, H, W = imgs.shape

            if images_chw is None:
                images_chw = (C, H, W)

            assert images_chw == (C, H, W)

            image_feats = self.encode_imgs(imgs)  # BTxHWxD
            all_img_features[sensor] = image_feats

        concatenated_feats = []

        for k in self.visual_sensors:
            corresponding_camera_token = getattr(self, f'visual_sensor_token_{k}')
            concatenated_feats.append(all_img_features[k] + corresponding_camera_token)

        concatenated_feats = torch.cat(concatenated_feats, dim=1)

        if text_feats is None:
            text_feats = self.encode_text(goals)  # BxLxD
        B, L, D = text_feats.shape
        text_feats_ = text_feats.unsqueeze(1).tile(1, T, 1, 1).reshape(B * T, L, D)
        fusion_token = self.fusion_token.reshape(1, 1, D).tile(B * T, 1, 1)
        return fusion_token, concatenated_feats, text_feats, text_feats_, B, T, D

    def forward(
        self,
        frames,
        goals,
        text_feats=None,
        non_visual_sensors=None,
    ):
        (
            fusion_token,
            concatenated_feats,
            text_feats,
            text_feats_,
            B,
            T,
            D,
        ) = self.get_image_text_feats(frames, goals, text_feats)
        input_features = [fusion_token, concatenated_feats, text_feats_]

        fused_feats = self.fusion_xformer(torch.cat(input_features, 1))

        fused_feats = fused_feats[:, 0, :]  # BTxD

        return fused_feats.reshape(B, T, D), text_feats


class PositionalEncoder(nn.Module):
    def __init__(self, d_model: int, max_len: int = 1000):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        self.register_buffer('div_term', div_term)

    def forward(self, position):
        """
        Args:
            position: Tensor, shape [batch_size, seq_len]
        """
        B, L = position.shape
        position = position.unsqueeze(-1)  # BxLx1
        pe = torch.zeros([B, L, self.d_model], device=position.device)
        pe[:, :, 0::2] = torch.sin(position * self.div_term)
        pe[:, :, 1::2] = torch.cos(position * self.div_term)
        return pe
