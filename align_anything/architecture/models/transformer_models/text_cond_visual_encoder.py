# Copyright 2024 Allen Institute for AI

# Copyright 2024-2025 Align-Anything Team. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================


import math

# from utils.transformation_util import get_full_transformation_list, sample_a_specific_transform
from dataclasses import dataclass, field
from typing import List, Literal

import torch
import torch.nn as nn
from open_clip import create_model_from_pretrained
from open_clip.transformer import TextTransformer
from transformers import T5EncoderModel

from align_anything.architecture.models.transformer_models.image_encoders import IMAGE_ENCODERS
from align_anything.utils.utils.sensor_constant_utils import is_a_visual_sensor


@dataclass
class TransformerConfig:
    num_layers: int = 3
    d_model: int = 512
    nhead: int = 8


TEXT_ENCODER_DIMS = {
    't5-small': 512,
    't5-base': 768,
    't5-large': 1024,
    'SigLIPBase': 768,
    'SigLIPBase384': 768,
    'SigLIPBase384Resize': 768,
    'SigLIPLarge': 1024,
}


def create_text_encoder(encoder_name):
    if 'siglip' in encoder_name.lower():
        _, cfg = IMAGE_ENCODERS[encoder_name]
        encoder = create_model_from_pretrained(f'hf-hub:timm/{cfg.model}')[0].text
        encoder.output_tokens = True
        return encoder
    elif 't5' in encoder_name.lower():
        return T5EncoderModel.from_pretrained(encoder_name)
    else:
        raise NotImplementedError('Only SigLIP and T5 text encoders are supported.')


@dataclass
class TextCondVisualEncoderConfig:
    image_encoder: str = 'Dinov2Small'
    text_encoder: str = 't5-small'
    fusion_xformer: TransformerConfig = field(default_factory=lambda: TransformerConfig(3, 512, 8))
    input_sensors: List[str] = None
    bbox_encoding_type: Literal['positional'] = 'positional'


class TextCondMultiCameraVisualEncoder(nn.Module):
    def __init__(self, cfg: TextCondVisualEncoderConfig):
        super().__init__()
        self.cfg = cfg

        # TODO KE: This is just a hack to be backward compatible
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

        if 'task_relevant_object_bbox' in cfg.input_sensors:
            if self.cfg.bbox_encoding_type == 'positional':
                self.bbox_pos_encoder = nn.Sequential(
                    PositionalEncoder(32),
                    nn.Linear(32, self.cfg.fusion_xformer.d_model),
                    nn.LayerNorm(self.cfg.fusion_xformer.d_model),
                    nn.ReLU(),
                )
                self.coord_pos_enc = nn.Embedding(5, self.cfg.fusion_xformer.d_model)
            else:
                raise NotImplementedError(
                    f"Unknown bbox encoding type '{self.cfg.bbox_encoding_type}' for bbox sensor, "
                    f"must be one of ['positional']"
                )
        if 'manip_task_relevant_object_box' in cfg.input_sensors:
            if self.cfg.bbox_encoding_type == 'positional':
                self.manip_bbox_pos_encoder = nn.Sequential(
                    PositionalEncoder(32),
                    nn.Linear(32, self.cfg.fusion_xformer.d_model),
                    nn.LayerNorm(self.cfg.fusion_xformer.d_model),
                    nn.ReLU(),
                )
                self.manip_coord_pos_enc = nn.Embedding(5, self.cfg.fusion_xformer.d_model)
            else:
                raise NotImplementedError(
                    f"Unknown bbox encoding type '{self.cfg.bbox_encoding_type}' for bbox sensor, "
                    f"must be one of ['positional']"
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
            self.image_encoder(imgs.reshape(B * T, C, H, W))  # shape: [496, 384, 7, 12]
        )  # BTxC_xH_xW_
        _, C_, H_, W_ = feats.shape
        feats = feats.reshape(B * T, C_, H_ * W_).permute(
            0, 2, 1
        )  # BTxH_W_xC_ ([496, 84, 512]) - make sense
        return self.visual_adapter(feats)  # doesn't change shape

    def create_compressor(self):
        return nn.Sequential(
            nn.Conv2d(self.image_encoder.cfg.output_size[0], self.cfg.fusion_xformer.d_model, 1),
            nn.ReLU(),
            nn.Conv2d(self.cfg.fusion_xformer.d_model, self.cfg.fusion_xformer.d_model, 1),
            nn.ReLU(),
        )

    def forward(
        self,
        frames,
        goals,
        text_feats=None,
        task_relevant_object_bbox=None,
        manip_task_relevant_object_bbox=None,
    ):
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

        # KE: I checked the gradients on all the values.

        input_features = [fusion_token, concatenated_feats, text_feats_]

        if task_relevant_object_bbox is not None:
            B, T, N = task_relevant_object_bbox.shape
            if self.cfg.bbox_encoding_type == 'positional':
                task_relevant_object_bbox = task_relevant_object_bbox.reshape(B * T, N)
                bbox_feats = self.bbox_pos_encoder(task_relevant_object_bbox)
                bbox_feats = bbox_feats + self.coord_pos_enc(
                    torch.tensor([[0, 1, 2, 3, 4]], device=bbox_feats.device).tile(B * T, 1)
                )
            else:
                raise NotImplementedError

            input_features.append(bbox_feats)

        if manip_task_relevant_object_bbox is not None:
            B, T, N = manip_task_relevant_object_bbox.shape
            if self.cfg.bbox_encoding_type == 'positional':
                manip_task_relevant_object_bbox = manip_task_relevant_object_bbox.reshape(B * T, N)
                bbox_feats = self.manip_bbox_pos_encoder(manip_task_relevant_object_bbox)
                bbox_feats = bbox_feats + self.manip_coord_pos_enc(
                    torch.tensor([[0, 1, 2, 3, 4]], device=bbox_feats.device).tile(B * T, 1)
                )
            else:
                raise NotImplementedError

            input_features.append(bbox_feats)

        # TODO KIANA WE NEED TO HAVE ASSERTIONS
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


@dataclass
class NonTxVisualEncoderConfig:
    image_encoder: str = 'Dinov2Small'
    text_encoder: str = 't5-small'
    input_sensors: List[str] = None
    compressor_hidden_dims: List[int] = (128, 32)
    text_adapter_output_dim: int = 32
    image_text_combiner_hidden_dims: List[int] = (64, 32)
    per_cam_feat_dim: int = 2688
    final_out_dim: int = 512


class NonTxMultiCameraVisualEncoder(nn.Module):
    def __init__(self, cfg: NonTxVisualEncoderConfig):
        super().__init__()
        self.cfg = cfg

        # TODO KE: This is just a hack to be backward compatible
        if cfg.image_encoder == 'dinov2' and cfg.image_encoder not in IMAGE_ENCODERS:
            cfg.image_encoder = 'Dinov2Small'
            print('REAPLACING DINOV2 WITH DINOV2SMALL')

        if cfg.image_encoder in IMAGE_ENCODERS:
            image_encoder_model_cls, image_encoder_cfg = IMAGE_ENCODERS[cfg.image_encoder]
            self.image_encoder = image_encoder_model_cls(image_encoder_cfg)
        else:
            raise NotImplementedError()
        self.visual_compressor = self.create_compressor()

        self.text_encoder = create_text_encoder(cfg.text_encoder)

        # text_adapter maps T5/SigLIP text embeddings to the action decoders dimension for use as memory
        self.text_adapter = nn.Sequential(
            nn.Linear(TEXT_ENCODER_DIMS[cfg.text_encoder], self.cfg.final_out_dim),
            nn.LayerNorm(self.cfg.final_out_dim),
            nn.ReLU(),
        )
        # text_adapter_for_combiner maps the text embedding (after text_adapter) to the dimension required for image text combination
        self.text_adapter_for_combiner = nn.Sequential(
            nn.Linear(self.cfg.final_out_dim, self.cfg.text_adapter_output_dim),
            nn.LayerNorm(self.cfg.text_adapter_output_dim),
            nn.ReLU(),
        )

        self.image_text_combiner = self.create_image_text_combiner()
        self.visual_sensors = [sensor for sensor in cfg.input_sensors if is_a_visual_sensor(sensor)]
        self.final_adapter = nn.Sequential(
            nn.Linear(len(self.visual_sensors) * 32 * 7 * 12, self.cfg.final_out_dim),
            nn.LayerNorm(self.cfg.final_out_dim),
            nn.ReLU(),
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
        return feats

    def create_compressor(self):
        assert len(self.cfg.compressor_hidden_dims) == 2
        return nn.Sequential(
            nn.Conv2d(self.image_encoder.cfg.output_size[0], self.cfg.compressor_hidden_dims[0], 1),
            nn.ReLU(),
            nn.Conv2d(self.cfg.compressor_hidden_dims[0], self.cfg.compressor_hidden_dims[1], 1),
            nn.ReLU(),
        )

    def create_image_text_combiner(self):
        assert len(self.cfg.image_text_combiner_hidden_dims) == 2
        return nn.Sequential(
            nn.Conv2d(
                self.cfg.compressor_hidden_dims[-1] + self.cfg.text_adapter_output_dim,
                self.cfg.image_text_combiner_hidden_dims[0],
                1,
            ),
            nn.ReLU(),
            nn.Conv2d(
                self.cfg.image_text_combiner_hidden_dims[0],
                self.cfg.image_text_combiner_hidden_dims[1],
                1,
            ),
            nn.ReLU(),
        )

    def forward(
        self,
        frames,
        goals,
        text_feats=None,
        task_relevant_object_bbox=None,
        manip_task_relevant_object_bbox=None,
    ):
        assert task_relevant_object_bbox is None and manip_task_relevant_object_bbox is None

        all_img_features = {}
        images_chw = None
        for sensor in frames.keys():
            assert is_a_visual_sensor(sensor)
            imgs = frames[sensor]
            B, T, C, H, W = imgs.shape

            if images_chw is None:
                images_chw = (C, H, W)

            assert images_chw == (C, H, W)

            image_feats = self.encode_imgs(imgs)  # BTxCxHxW
            all_img_features[sensor] = image_feats

            _, fC, fH, fW = image_feats.shape

        if text_feats is None:
            text_feats = self.encode_text(goals)  # BxLxD

        text_feats_ = self.text_adapter_for_combiner(text_feats)
        text_feats_ = text_feats_.mean(dim=1, keepdim=True).tile(1, T, 1).reshape(B * T, -1)  # BTxD
        text_feats_ = text_feats_.unsqueeze(-1).unsqueeze(-1).tile(1, 1, fH, fW)  # BTxDxHxW

        all_cam_feats = []
        for sensor in frames.keys():
            all_cam_feats.append(
                self.image_text_combiner(
                    torch.cat([all_img_features[sensor], text_feats_], dim=1)
                ).reshape(B, T, -1)
            )

        fused_feats = self.final_adapter(torch.cat(all_cam_feats, dim=-1))
        return fused_feats, text_feats
