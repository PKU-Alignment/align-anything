# Copyright 2024 Allen Institute for AI

# Copyright 2025 Align-Anything Team. All Rights Reserved.
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
from typing import Any, Dict, List, Optional, Sequence, Union, cast

import gym
import numpy as np
import torch
import torch.nn as nn
from allenact.base_abstractions.preprocessor import Preprocessor
from allenact.utils.misc_utils import prepare_locals_for_super

from align_anything.utils.spoc_utils.transformation_util import (
    get_transformation,
    sample_a_specific_transform,
)


DINO_PRETRAINED_MODEL = ['dinov2_vits14', 'dinov2_vitb14', 'dinov2_vitl14', 'dinov2_vitg14']


class DinoViTEmbedder(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model
        self.pool = nn.AdaptiveAvgPool2d((7, 12))
        self.eval()

    def forward(self, x):
        assert x.shape[-2:] == (224, 384), f'Expected shape is 224x384; got {x.shape}'
        with torch.no_grad():
            x = self.model.forward_features(x[:, :, :, 3:-3])['x_norm_patchtokens']
            B, _, D = x.shape  # Bx432x384
            x = x.permute(0, 2, 1)  # Bx384x432
            x = x.reshape(B, D, 16, 27)
            x = self.pool(x)
            return x


class DinoViTPreprocessor(Preprocessor):
    DINO_RGB_MEANS = (0.48145466, 0.4578275, 0.40821073)
    DINO_RGB_STDS = (0.26862954, 0.26130258, 0.27577711)

    def __init__(
        self,
        rgb_input_uuid: str,
        dino_model_type: str,
        device: Optional[torch.device] = None,
        device_ids: Optional[List[torch.device]] = None,
        flatten: bool = True,
        **kwargs: Any,
    ):
        assert dino_model_type in DINO_PRETRAINED_MODEL

        if dino_model_type == 'dinov2_vits14':
            if flatten:
                output_shape = (7 * 12, 384)
            else:
                output_shape = (7, 12, 384)
        elif dino_model_type == 'dinov2_vitb14':
            if flatten:
                output_shape = (7 * 12, 768)
            else:
                output_shape = (7, 12, 768)
        elif dino_model_type == 'dinov2_vitl14':
            if flatten:
                output_shape = (7 * 12, 1024)
            else:
                output_shape = (7, 12, 1024)
        elif dino_model_type == 'dinov2_vitg14':
            if flatten:
                output_shape = (7 * 12, 1536)
            else:
                output_shape = (7, 12, 1536)
        else:
            raise NotImplementedError(
                f"Currently `dino_model_type` must be one of 'dinov2_vits14', 'dinov2_vitb14', 'dinov2_vitl14', or 'dinov2_vitg14'"
            )

        self.dino_model_type = dino_model_type

        self.device = torch.device('cpu') if device is None else device
        self.device_ids = device_ids or cast(
            List[torch.device], list(range(torch.cuda.device_count()))
        )
        self._vit: Optional[DinoViTEmbedder] = None

        low = -np.inf
        high = np.inf
        shape = output_shape

        input_uuids = [rgb_input_uuid]
        assert len(input_uuids) == 1, 'resnet preprocessor can only consume one observation type'

        observation_space = gym.spaces.Box(low=low, high=high, shape=shape)

        super().__init__(**prepare_locals_for_super(locals()))

    @property
    def vit(self) -> DinoViTEmbedder:
        if self._vit is None:
            self._vit = DinoViTEmbedder(
                model=torch.hub.load('facebookresearch/dinov2', self.dino_model_type),
            ).to(self.device)
            for module in self._vit.modules():
                if 'BatchNorm' in type(module).__name__:
                    module.momentum = 0.0
            self._vit.eval()
        return self._vit

    def to(self, device: torch.device) -> 'DinoViTPreprocessor':
        self._vit = self.vit.to(device)
        self.device = device
        return self

    def process(self, obs: Dict[str, Any], *args: Any, **kwargs: Any) -> Any:
        x = obs[self.input_uuids[0]].to(self.device).permute(0, 3, 1, 2)  # bhwc -> bchw
        # If the input is depth, repeat it across all 3 channels
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        x = self.vit(x).float()
        return x


class PostPreprocessor(nn.Module):
    def __init__(self, inp_dim, h_dim=384, preserve_spatial=True):
        super().__init__()
        self.preserve_spatial = preserve_spatial
        if not preserve_spatial:
            self.post_process = nn.Sequential(
                nn.Conv2d(inp_dim, h_dim, 1),
                nn.ReLU(),
                nn.Conv2d(h_dim, h_dim, (3, 5)),
                nn.ReLU(),
                nn.Conv2d(h_dim, h_dim, (3, 5)),
                nn.ReLU(),
                nn.Conv2d(h_dim, h_dim, (3, 4)),
                nn.ReLU(),
                nn.Conv2d(h_dim, h_dim, 1),
                nn.ReLU(),
                nn.Flatten(start_dim=2),
            )
        else:
            self.post_process = nn.Sequential(nn.Flatten(start_dim=2))

    def forward(self, x):
        t = 0
        if len(x.shape) > 4:
            b, t, f, h, w = x.shape
            x = torch.reshape(x, (b * t, f, h, w))
        x = self.post_process(x)
        if t != 0:
            if not self.preserve_spatial:
                x = torch.reshape(x, (b, t, -1))
            else:
                x = torch.permute(torch.reshape(x, (b, t, f, h * w)), (0, 1, 3, 2))
        elif self.preserve_spatial:
            x = torch.permute(x, (0, 2, 1))
        return x


# this is always called for allenact models
class DataAugmentationPreprocessor(Preprocessor):
    def __init__(
        self,
        rgb_input_uuid: str,
        device: Optional[torch.device] = None,
        device_ids: Optional[List[torch.device]] = None,
        normalize: bool = True,
        mean: Optional[Union[np.ndarray, Sequence[float]]] = DinoViTPreprocessor.DINO_RGB_MEANS,
        stdev: Optional[Union[np.ndarray, Sequence[float]]] = DinoViTPreprocessor.DINO_RGB_STDS,
        height: Optional[int] = None,
        width: Optional[int] = None,
        output_channels: Optional[int] = 3,
        num_steps_to_change: int = 500,
        use_augmentation: bool = True,
        **kwargs: Any,
    ):
        assert height is not None and width is not None
        self.device = torch.device('cpu') if device is None else device
        self.device_ids = device_ids or cast(
            List[torch.device], list(range(torch.cuda.device_count()))
        )
        self._vit: Optional[DinoViTEmbedder] = None

        self.normalize = normalize
        self.mean = torch.from_numpy(np.array(mean, dtype=np.float32).reshape((1, len(mean), 1, 1)))
        self.stdev = torch.from_numpy(
            np.array(stdev, dtype=np.float32).reshape((1, len(stdev), 1, 1))
        )
        self.num_steps_to_change = num_steps_to_change
        self.num_steps = 0
        self.augmentations = None
        self.use_augmentation = use_augmentation

        low = -np.inf
        high = np.inf
        shape = (cast(int, height), cast(int, width), cast(int, output_channels))

        input_uuids = [rgb_input_uuid]
        assert len(input_uuids) == 1, 'resnet preprocessor can only consume one observation type'

        observation_space = gym.spaces.Box(low=low, high=high, shape=shape)

        super().__init__(**prepare_locals_for_super(locals()))

    def to(self, device: torch.device) -> 'DataAugmentationPreprocessor':
        self.mean = self.mean.to(device)
        self.stdev = self.stdev.to(device)
        self.device = device
        return self

    def process(self, obs: Dict[str, Any], *args: Any, **kwargs: Any) -> Any:
        if self.use_augmentation:
            if self.num_steps == 0 or self.augmentations is None:
                self.augmentations = sample_a_specific_transform(get_transformation())
            self.num_steps = (self.num_steps + 1) % self.num_steps_to_change
            x = self.augmentations(obs[self.input_uuids[0]].permute(0, 3, 1, 2)).to(
                self.device
            )  # bhwc -> bchw
        else:
            x = obs[self.input_uuids[0]].permute(0, 3, 1, 2).to(self.device)
        x = x.float() / 255.0
        if self.normalize:
            x -= self.mean
            x /= self.stdev
        x = x.permute(0, 2, 3, 1)  # bchw -> bhwc
        return x
