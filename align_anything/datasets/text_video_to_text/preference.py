# Copyright 2024 PKU-Alignment Team and LlamaFactory team. All Rights Reserved.
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

from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, List, Optional, Sequence, Tuple, TypedDict, Union, Any, Callable
from typing_extensions import TypedDict  # Python 3.10+

import torch
import transformers
from torch.utils.data import Dataset
from torchvision import transforms
from transformers.tokenization_utils import PaddingStrategy, TruncationStrategy
from transformers import PreTrainedTokenizer, ProcessorMixin, DataCollatorForSeq2Seq
from transformers.image_processing_utils import BaseImageProcessor

from align_anything.utils.multi_process import get_current_device
from align_anything.utils.template_registry import get_template_class
from align_anything.utils.tools import right_padding
from datasets import load_dataset
from copy import deepcopy
from io import BytesIO
import numpy as np
import av
from PIL import Image
from PIL.Image import Image as ImageObject

class EncodedImage(TypedDict):
    path: Optional[str]
    bytes: Optional[bytes]
ImageInput = Union[str, EncodedImage, ImageObject]
VideoInput = str


IGNORE_INDEX = -100
IMAGE_PLACEHOLDER = "<image>"
VIDEO_PLACEHOLDER = "<video>"

__all__ = [
    'PreferenceDataset',
    'PreferenceCollator',
    'PreferenceSample',
    'PreferenceBatch',
]


class PreferenceSample(TypedDict, total=True):
    input_ids: torch.LongTensor  # size = (L,)
    labels: torch.LongTensor  # size = (L,)
    pixel_values: torch.LongTensor | None  # size = (B, C, H, W)


class PreferenceBatch(TypedDict, total=True):
    input_ids: torch.LongTensor  # size = (B, L)
    labels: torch.LongTensor  # size = (B, L)
    attention_mask: torch.BoolTensor  # size = (B, L)
    pixel_values: torch.LongTensor | None  # size = (B, C, H, W)

def _regularize_images(
    images: Sequence["ImageInput"],
    processor: "ProcessorMixin",
    max_resolution: Optional[int] = None,
) -> List["ImageObject"]:
    r"""
    Regularizes images to avoid error. Including reading, resizing and converting.
    """
    if max_resolution is None:
        max_resolution: int = getattr(processor, "image_resolution", 512)

    results = []
    for image in images:
        if isinstance(image, str):
            image = Image.open(image)
        elif isinstance(image, dict):
            if image["bytes"] is not None:
                image = Image.open(BytesIO(image["bytes"]))
            else:
                image = Image.open(image["path"])

        if not isinstance(image, ImageObject):
            raise ValueError("Expect input is a list of Images, but got {}.".format(type(image)))

        if max(image.width, image.height) > max_resolution:
            factor = max_resolution / max(image.width, image.height)
            image = image.resize((int(image.width * factor), int(image.height * factor)), resample=Image.NEAREST)

        if image.mode != "RGB":
            image = image.convert("RGB")

        results.append(image)

    return results


def _regularize_videos(
    videos: Sequence["VideoInput"],
    processor: "ProcessorMixin",
) -> List[List["ImageObject"]]:
    r"""
    Regularizes videos to avoid error. Including reading, resizing and converting.
    """
    video_resolution: int = getattr(processor, "video_resolution", 128)
    video_fps: float = getattr(processor, "video_fps", 1.0)
    video_maxlen: int = getattr(processor, "video_maxlen", 64)
    video_factor: int = getattr(processor, "video_factor", 2)
    results = []
    for video in videos:
        container = av.open(video, "r")
        video_stream = next(stream for stream in container.streams if stream.type == "video")
        total_frames = video_stream.frames
        sample_frames = float(video_stream.duration * video_stream.time_base) * video_fps
        sample_frames = min(video_maxlen, sample_frames)  # reduce length <= maxlen
        sample_frames = round(sample_frames / video_factor) * video_factor  # for qwen2_vl
        sample_indices = np.linspace(0, total_frames - 1, sample_frames).astype(np.int32)
        frames: List["ImageObject"] = []
        container.seek(0)
        for frame_idx, frame in enumerate(container.decode(video_stream)):
            if frame_idx in sample_indices:
                frames.append(frame.to_image())

        frames = _regularize_images(frames, processor, video_resolution)
        results.append(frames)
        
    return results

def _get_mm_inputs(
    images: Sequence["ImageInput"],
    videos: Sequence["VideoInput"],
    processor: "ProcessorMixin",
) -> Dict[str, "torch.Tensor"]:
    r"""
    Processes visual inputs.

    Returns: (llava and paligemma)
        pixel_values: tensor with shape (B, C, H, W)

    Returns: (qwen2-vl)
        pixel_values: tensor with shape (num_patches, patch_dim)
        image_grid_thw: tensor with shape (num_images, 3), where the three numbers are time, width, height

    It holds num_patches == torch.prod(image_grid_thw)
    """
    image_processor: "BaseImageProcessor" = getattr(processor, "image_processor")
    input_dict = {"images": None}  # default key
    if len(images) != 0:
        images = _regularize_images(images, processor)
        input_dict["images"] = images

    if len(videos) != 0:
        videos = _regularize_videos(videos, processor)
        input_dict["videos"] = videos

    if input_dict.get("images", None) is not None or input_dict.get("videos", None) is not None:
        return image_processor(**input_dict, return_tensors="pt")
    else:
        return {}

def process_messages(
    message: str,
    images: Sequence["ImageInput"],
    videos: Sequence["VideoInput"],
    processor: Optional["ProcessorMixin"],
) -> List[Dict[str, str]]:
    image_processor: "BaseImageProcessor" = getattr(processor, "image_processor")
    merge_length: int = getattr(image_processor, "merge_size") ** 2
    mm_inputs = _get_mm_inputs(images, videos, processor)
    image_grid_thw = mm_inputs.get("image_grid_thw", [])
    video_grid_thw = mm_inputs.get("video_grid_thw", [])

    num_image_tokens, num_video_tokens = 0, 0
    
    content = message
    while IMAGE_PLACEHOLDER in content:
        if num_image_tokens >= len(image_grid_thw):
            raise ValueError("`len(images)` is less than the number of {} tokens.".format(IMAGE_PLACEHOLDER))

        content = content.replace(
            IMAGE_PLACEHOLDER,
            "<|vision_start|>{}<|vision_end|>".format(
                "<|image_pad|>" * (image_grid_thw[num_image_tokens].prod() // merge_length)
            ),
            1,
        )
        num_image_tokens += 1

    while VIDEO_PLACEHOLDER in content:
        if num_video_tokens >= len(video_grid_thw):
            raise ValueError("`len(videos)` is less than the number of {} tokens.".format(VIDEO_PLACEHOLDER))

        content = content.replace(
            VIDEO_PLACEHOLDER,
            "<|vision_start|>{}<|vision_end|>".format(
                "<|video_pad|>" * (video_grid_thw[num_video_tokens].prod() // merge_length)
            ),
            1,
        )
        num_video_tokens += 1

    return_message = content

    if len(images) != num_image_tokens:
        raise ValueError("The number of images does not match the number of {} tokens".format(IMAGE_PLACEHOLDER))

    if len(videos) != num_video_tokens:
        raise ValueError("The number of videos does not match the number of {} tokens".format(VIDEO_PLACEHOLDER))
    
    return return_message

class PreferenceDataset(Dataset):

    def __init__(
        self,
        path: str,
        template: str,
        tokenizer: transformers.PreTrainedTokenizer,
        processor: transformers.ProcessorMixin | transforms.Compose | None = None,
        name: str | None = None,
        size: int | None = None,
        split: str | None = None,
        subset: str | None = None,
        data_files: str | None = None,
        optional_args: list | str = [],
    ):
        super().__init__()
        assert path, f'You must set the valid datasets path! Here is {path}'
        assert template, f'You must set the valid template path! Here is {template}'
        self.path = path
        self.tokenizer = tokenizer
        self.processor = processor
        self.template = get_template_class(template)
        self.raw_data = load_dataset(
            path,
            name=name if name and name!="None" else None,
            split=split if split and split!="None" else None,
            data_files=data_files if data_files and data_files!="None" else None,
            *optional_args,
            trust_remote_code=True,
        )
        self.valid_indices = self.fillter_indices()
        if size:
            self.raw_data = self.raw_data.select(range(int(size)))
        

    
    def preprocess(self, raw_sample: dict[str, Any]) -> PreferenceSample:
        formatted_sample = self.template.format_preference_sample(raw_sample)
        return_dict = {}
        
        better_text = formatted_sample['better_text']
        worse_text = formatted_sample['worse_text']
        image = formatted_sample['image']
        video = formatted_sample['video']
        
        processed_better_text = process_messages(better_text, image, video, self.processor)
        processed_worse_text = process_messages(worse_text, image, video, self.processor)

        better_input_ids = self.tokenizer(processed_better_text, add_special_tokens=False, return_tensors='pt')['input_ids'][0]
        worse_input_ids = self.tokenizer(processed_worse_text, add_special_tokens=False, return_tensors='pt')['input_ids'][0]

        
        return_dict['better_input_ids'] = better_input_ids
        return_dict['worse_input_ids'] = worse_input_ids
        return_dict['images'] = image
        return_dict['videos'] = video
        
        return return_dict

    def fillter_indices(self):
        valid_indices = []
        for i, item in enumerate(self.raw_data):
            if not self.template.check_equal(item):
                valid_indices.append(i)
        return valid_indices

    def get_collator(self) -> Callable[[list[dict[str, torch.Tensor]]], dict[str, torch.Tensor]]:
        return PreferenceCollator(self.tokenizer, processor=self.processor)

    def tokenize(
        self,
        text: str,
        add_special_tokens: bool = True,
        padding: bool | str | PaddingStrategy = PaddingStrategy.DO_NOT_PAD,
        truncation: bool | str | TruncationStrategy = TruncationStrategy.LONGEST_FIRST,
        max_length: int | None = None,
    ) -> torch.LongTensor:  # size = (L,)
        """Tokenize a text string into a tensor representation."""
        if max_length is None:
            max_length = self.tokenizer.model_max_length

        return self.tokenizer(
            text,
            add_special_tokens=add_special_tokens,
            padding=padding,
            max_length=max_length,
            truncation=truncation,
            return_tensors='pt',
        )['input_ids'][0]

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        """Get a tokenized data sample by index."""
        raw_sample = self.raw_data[self.valid_indices[index]]
        data = self.preprocess(raw_sample.copy())
        return data

    def __len__(self) -> int:
        """Get the number of samples in the dataset."""
        return len(self.valid_indices)

@dataclass
class PreferenceCollator(DataCollatorForSeq2Seq):

    processor: Optional["ProcessorMixin"] = None

    def __init__(self, tokenizer: PreTrainedTokenizer, processor: Optional["ProcessorMixin"] = None):
        super().__init__(tokenizer)
        self.processor = processor

    def __call__(self, samples: Sequence[Dict[str, Any]])  -> Dict[str, "torch.Tensor"]:
        better_batch, worse_batch = [], []
        for sample in samples:
            better_batch.append({
                "input_ids": sample['better_input_ids'],
                "images": sample['images'],
                "videos": sample['videos']
            })
            worse_batch.append({
                "input_ids": sample['worse_input_ids'],
                "images": sample['images'],
                "videos": sample['videos']
            })
        
        total_batch = better_batch + worse_batch
        batch_images, batch_videos = [], []
        current_device = get_current_device()
        for sample in total_batch:
            images = sample.pop('images')
            videos = sample.pop('videos')
            batch_images.extend(images)
            batch_videos.extend(videos)

        mm_inputs = _get_mm_inputs(batch_images, batch_videos, self.processor)
        
        features: Dict[str, "torch.Tensor"] = super().__call__(total_batch)
        features.update(mm_inputs)

        return_dict = {}
        for k, v in features.items():
            if v is not None:
                return_dict[k] = v.to(current_device)
        return return_dict
