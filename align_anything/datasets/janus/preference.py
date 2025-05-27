# Copyright 2025 PKU-Alignment Team. All Rights Reserved.
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

from math import e
from einops import rearrange
from typing import Any, Callable
from typing_extensions import TypedDict  # Python 3.10+
import torch
import transformers
from torch.utils.data import Dataset
from torchvision import transforms
from transformers.tokenization_utils import PaddingStrategy, TruncationStrategy

from align_anything.utils.multi_process import get_current_device
from align_anything.utils.tools import right_padding, ends_with_any
from datasets import load_dataset


IGNORE_INDEX = -100

__all__ = [
    'PreferenceDataset',
    'PreferenceTokenizedDataset',
    'PreferenceCollator',
    'PreferenceSample',
    'PreferenceBatch',
]


class PreferenceSample(TypedDict, total=True):
    better_input_ids: torch.LongTensor  # size = (L,)
    worse_input_ids: torch.LongTensor  # size = (L,)
    pixel_values: torch.LongTensor | None  # size = (B, C, H, W)


class PreferenceBatch(TypedDict, total=True):
    better_input_ids: torch.LongTensor  # size = (B, L)
    worse_input_ids: torch.LongTensor  # size = (B, L)
    attention_mask: torch.BoolTensor  # size = (B, L)
    pixel_values: torch.LongTensor | None  # size = (B, C, H, W)
    task: str


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
        self.tokenizer = tokenizer
        self.processor = processor
        self.template = template

        if isinstance(optional_args, str):
            optional_args = [optional_args]
        self.raw_data = load_dataset(
            path,
            split=split,
            data_files=data_files,
            subset=subset,
            *optional_args,
            trust_remote_code=True,
        )
        if size:
            size = min(size, len(self.raw_data))
            self.raw_data = self.raw_data.select(range(int(size)))

    def preprocess(self, raw_sample: dict[str, Any]) -> PreferenceSample:
        better_conversation, worse_conversation, meta_info = self.template.format_preference_sample(raw_sample)

        if not ends_with_any(better_conversation, self.tokenizer.eos_token):
            better_conversation += self.tokenizer.eos_token

        if not ends_with_any(worse_conversation, self.tokenizer.eos_token):
            worse_conversation += self.tokenizer.eos_token
        
        better_inputs = self.processor(
            prompt=better_conversation, images=[meta_info['image']], return_tensors='pt'
        )
        worse_inputs = self.processor(
            prompt=worse_conversation, images=[meta_info['image']], return_tensors='pt'
        )
        
        return_dict = {}
        return_dict['better_input_ids'] = better_inputs['input_ids'][0]
        return_dict['worse_input_ids'] = worse_inputs['input_ids'][0]
        return_dict['pixel_values'] = better_inputs['pixel_values'][0]
        return_dict['better_images_seq_mask'] = better_inputs['images_seq_mask'][0]
        return_dict['worse_images_seq_mask'] = worse_inputs['images_seq_mask'][0]
        return_dict['better_images_emb_mask'] = better_inputs['images_emb_mask'][0]
        return_dict['worse_images_emb_mask'] = worse_inputs['images_emb_mask'][0]
        return_dict['task'] = 'understanding'

        return return_dict

    def get_collator(self) -> Callable[[list[dict[str, torch.Tensor]]], dict[str, torch.Tensor]]:
        return PreferenceCollator(self.tokenizer.pad_token_id)

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
        raw_sample = self.raw_data[index]
        data = self.preprocess(raw_sample.copy())
        return data

    def __len__(self) -> int:
        """Get the number of samples in the dataset."""
        return len(self.raw_data)


class PreferenceTokenizedDataset(Dataset):

    def __init__(
        self,
        path: str,
        template: str | None = None,
        tokenizer: transformers.PreTrainedTokenizer | None = None,
        processor: transformers.ProcessorMixin | transforms.Compose | None = None,
        size: int | None = None,
        name: str | None = None,
        split: str | None = None,
        subset: str | None = None,
        data_files: str | None = None,
        optional_args: list | str = [],
    ):
        super().__init__()
        assert path, f'You must set the valid datasets path! Here is {path}'
        assert template, f'You must set the valid template path! Here is {template}'
        self.tokenizer = tokenizer
        self.processor = processor

        self.raw_data = torch.load(f'{path}/{data_files}', map_location=torch.device('cpu'))
        if size:
            self.raw_data = self.raw_data.select(range(int(size)))
        self.template = template

    def get_collator(self) -> Callable[[list[dict[str, torch.Tensor]]], dict[str, torch.Tensor]]:
        return PreferenceCollator(self.tokenizer.pad_token_id)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        """Get a tokenized data sample by index."""
        raw_sample = self.raw_data[index]
        return raw_sample

    def __len__(self) -> int:
        """Get the number of samples in the dataset."""
        return len(self.raw_data)


class PreferenceCollator:

    def __init__(self, pad_token_id: int) -> None:
        """Initialize a collator."""
        self.pad_token_id = pad_token_id

    def __call__(self, samples: list[PreferenceSample]) -> PreferenceBatch:
        return_dict = {}
        current_device = get_current_device()

        input_ids = [sample['better_input_ids'] for sample in samples] + [
            sample['worse_input_ids'] for sample in samples
        ]  # size = (2 * B, L)
        return_dict['input_ids'] = right_padding(input_ids, padding_value=self.pad_token_id).to(
            current_device
        )  # size = (2 * B, L)

        if 'pixel_values' in samples[0].keys():
            pixel_values = [sample['pixel_values'] for sample in samples]
            return_dict['pixel_values'] = torch.cat(pixel_values + pixel_values, dim=0).to(current_device).unsqueeze(0)
        
        if "better_images_emb_mask" in samples[0].keys():
            better_images_emb_mask  = [sample['better_images_emb_mask'] for sample in samples]
            worse_images_emb_mask = [sample['worse_images_emb_mask'] for sample in samples]
            return_dict['images_emb_mask'] = right_padding(better_images_emb_mask + worse_images_emb_mask, padding_value=0).to(current_device).detach()
            
        return_dict['task'] = samples[0]['task']
        return return_dict
