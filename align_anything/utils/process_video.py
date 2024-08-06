# Copyright 2024 PKU-Alignment Team. All Rights Reserved.
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
"""The video processing utility."""

from typing import List, Optional, Union

import numpy as np
import PIL
import torch
from diffusers.video_processor import VideoProcessor
from PIL import Image


def get_video_processor(
    sample_frames: int, resolution: int, do_resize: bool = True, **kwargs
) -> VideoProcessor:
    class CustomVideoProcessor(VideoProcessor):
        def __init__(self, sample_frames: int, resolution: int, do_resize: bool = True, **kwargs):
            self.sample_frames = sample_frames
            self.resolution = resolution
            super().__init__(do_resize=do_resize, **kwargs)

        def extract_frames(self, frames: List[PIL.Image]) -> List[PIL.Image]:
            num_frames = len(frames)

            if num_frames >= 2 * self.sample_frames:
                selected_indices = np.linspace(
                    0, 2 * self.sample_frames - 1, self.sample_frames, dtype=int
                )
                extracted_frames = [frames[i] for i in selected_indices]
            elif num_frames >= self.sample_frames:
                selected_indices = np.linspace(0, num_frames - 1, self.sample_frames, dtype=int)
                extracted_frames = [frames[i] for i in selected_indices]
            else:
                selected_indices = np.linspace(0, num_frames - 1, self.sample_frames, dtype=int)
                extracted_frames = [frames[i % num_frames] for i in selected_indices]
            return extracted_frames

        def __call__(
            self,
            video: Optional[Union[torch.Tensor, List[PIL.Image], List[List[PIL.Image]]]],
            height: Optional[int] = None,
            width: Optional[int] = None,
        ) -> torch.Tensor:
            if height == None:
                height = self.resolution
            if width == None:
                width = self.resolution
            processed_video = None
            if isinstance(video, torch.Tensor):
                frames = [Image.fromarray(frame.numpy()) for frame in video]
                extract_frames = self.extract_frames(frames)
                processed_video = self.preprocess_video(extract_frames, height, width)
            else:
                processed_video = self.preprocess_video(video, height, width)
            if len(processed_video.shape) == 5 and processed_video.size(0) == 1:
                processed_video = processed_video.squeeze(0)
            return processed_video

    return CustomVideoProcessor(sample_frames, resolution, do_resize, **kwargs)
