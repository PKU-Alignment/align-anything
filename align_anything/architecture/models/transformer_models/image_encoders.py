from dataclasses import dataclass
from typing import Tuple

import clip
import torch
import torch.nn as nn
import torch.nn.functional as F
from open_clip import create_model_from_pretrained


@dataclass
class ClipResNetConfig:
    model: str = "RN50"
    pool: bool = False
    device: str = "cpu"
    output_size: Tuple[int, int, int] = (2048, 7, 12)


class ClipResNet(nn.Module):
    def __init__(self, cfg: ClipResNetConfig):  # pool=True, device='cpu'):
        super().__init__()
        self.cfg = cfg
        self.model = clip.load(cfg.model, device=torch.device(cfg.device), jit=False)[0]
        self.eval()

    def forward(self, x):
        m = self.model.visual
        with torch.no_grad():

            def stem(x):
                x = m.relu1(m.bn1(m.conv1(x)))
                x = m.relu2(m.bn2(m.conv2(x)))
                x = m.relu3(m.bn3(m.conv3(x)))
                x = m.avgpool(x)
                return x

            x = x.type(m.conv1.weight.dtype)
            x = stem(x)
            x = m.layer1(x)
            x = m.layer2(x)
            x = m.layer3(x)
            x = m.layer4(x)
            if self.cfg.pool:
                x = F.adaptive_avg_pool2d(x, (1, 1))
                x = torch.flatten(x, 1)

            return x


@dataclass
class Dinov2Config:
    model: str = "dinov2_vits14"
    output_size: Tuple[int, int, int] = (384, 7, 12)


class Dinov2(nn.Module):
    def __init__(self, cfg: Dinov2Config):
        super().__init__()
        self.cfg = cfg
        self.model = torch.hub.load("facebookresearch/dinov2", cfg.model)
        self.pool = nn.AdaptiveAvgPool2d(cfg.output_size[1:])
        self.eval()

    def forward(self, x):
        assert x.shape[-2:] == (224, 384), f"Expected shape is 224x384; got {x.shape}"
        with torch.no_grad():
            x = self.model.forward_features(x[:, :, :, 3:-3])["x_norm_patchtokens"]
            B, _, D = x.shape  # Bx432x384
            x = x.permute(0, 2, 1)  # Bx384x432
            x = x.reshape(B, D, 16, 27)
            x = self.pool(x)
            return x


@dataclass
class SigLIPConfig:
    model: str = "ViT-B-16-SigLIP-256"
    output_size: Tuple[int, int, int] = (768, 7, 12)


class SigLIP(nn.Module):
    def __init__(self, cfg: Dinov2Config):
        super().__init__()
        self.cfg = cfg
        siglip_full_model = create_model_from_pretrained("hf-hub:timm/{}".format(cfg.model))
        self.model = siglip_full_model[0].visual.trunk
        self.context_length = siglip_full_model[0].context_length
        self.pool = nn.AdaptiveAvgPool2d(cfg.output_size[1:])
        self.eval()

    def forward(self, x):
        assert x.shape[-2:] == (256, 256), f"Expected shape is 256x256; got {x.shape}"
        with torch.no_grad():
            x = self.model.forward_features(x)
            B, _, D = x.shape  # Bx256x768
            x = x.permute(0, 2, 1)  # Bx768x256
            x = x.reshape(B, D, 16, 16)
            x = self.pool(x)
            return x


IMAGE_ENCODERS = dict(
    Dinov2Small=(Dinov2, Dinov2Config()),
    Dinov2Base=(Dinov2, Dinov2Config(model="dinov2_vitb14", output_size=(768, 7, 12))),
    ClipResNet50=(ClipResNet, ClipResNetConfig()),
    SigLIPBase=(SigLIP, SigLIPConfig()),
    SigLIPLarge=(SigLIP, SigLIPConfig(model="ViT-L-16-SigLIP-256", output_size=(1024, 7, 12))),
)
