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

import os
from typing import Any, Callable
from typing_extensions import TypedDict  # Python 3.10+

import torch
import transformers
from torch.utils.data import Dataset
from torchvision import transforms
from transformers.tokenization_utils import PaddingStrategy, TruncationStrategy

from align_anything.utils.multi_process import get_current_device
from align_anything.utils.tools import right_padding, convert_to_rgb, ends_with_any
from datasets import load_dataset


IGNORE_INDEX = -100

__all__ = [
    'SupervisedDataset',
    'SupervisedTokenizedDataset',
    'SupervisedCollator',
    'SupervisedSample',
    'SupervisedBatch',
]


class SupervisedSample(TypedDict, total=True):
    input_ids: torch.LongTensor  # size = (L,)
    labels: torch.LongTensor  # size = (L,)
    pixel_values: torch.LongTensor | None  # size = (B, C, H, W)


class SupervisedBatch(TypedDict, total=True):
    input_ids: torch.LongTensor  # size = (B, L)
    labels: torch.LongTensor  # size = (B, L)
    attention_mask: torch.BoolTensor  # size = (B, L)
    pixel_values: torch.LongTensor | None  # size = (B, C, H, W)
    task: str
    images_seq_mask: torch.BoolTensor | None  # size = (B, L)
    images_emb_mask: torch.BoolTensor | None  # size = (B, N)


class SupervisedDataset(Dataset):

    def __init__(
        self,
        path: str,
        template: str,
        tokenizer: transformers.PreTrainedTokenizer,
        processor: transformers.ProcessorMixin | transforms.Compose | None = None,
        padding_side: str = 'right',
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
        self.padding_side = padding_side
        self.raw_data = load_dataset(
            path,
            split=split,
            data_files=data_files,
            subset=subset,
            *optional_args,
            trust_remote_code=True,
        )
        if size:
            self.raw_data = self.raw_data.select(range(int(size)))
        self.template = template

    def preprocess(self, raw_sample: dict[str, Any]) -> SupervisedSample:
        prompt, conversation, meta_info = self.template.format_supervised_sample(raw_sample)
        if not ends_with_any(conversation, self.tokenizer.eos_token):
            conversation += self.tokenizer.eos_token

        # return return_dict
        full_inputs = self.processor(
            prompt=conversation, images=[meta_info['image']], return_tensors='pt'
        )
        prompt_inputs = self.processor(
            prompt=prompt, images=[meta_info['image']], return_tensors='pt'
        )

        return_dict = {}
        return_dict['input_ids'] = full_inputs['input_ids'][0]
        return_dict['attention_mask'] = full_inputs['attention_mask'][0]
        return_dict['pixel_values'] = full_inputs['pixel_values'][0]
        return_dict['images_seq_mask'] = full_inputs['images_seq_mask'][0]
        return_dict['images_emb_mask'] = full_inputs['images_emb_mask'][0]
        return_dict['labels'] = return_dict['input_ids'].clone()
        return_dict['labels'][: len(prompt_inputs['input_ids'][0])] = IGNORE_INDEX
        return_dict['task'] = 'understanding'

        return return_dict

    def get_collator(self) -> Callable[[list[dict[str, torch.Tensor]]], dict[str, torch.Tensor]]:
        return SupervisedCollator(self.tokenizer.pad_token_id, self.processor, self.padding_side)

    def tokenize(
        self,
        conversation: str,
        add_special_tokens: bool = True,
        padding: bool | str | PaddingStrategy = PaddingStrategy.DO_NOT_PAD,
        truncation: bool | str | TruncationStrategy = TruncationStrategy.LONGEST_FIRST,
        max_length: int | None = None,
    ) -> torch.LongTensor:  # size = (L,)
        """Tokenize a text string into a tensor representation."""
        if max_length is None:
            max_length = self.tokenizer.model_max_length

        return self.tokenizer(
            text=conversation,
            add_special_tokens=add_special_tokens,
            padding=padding,
            max_length=max_length,
            truncation=truncation,
            return_tensors='pt',
        )

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        """Get a tokenized data sample by index."""
        raw_sample = self.raw_data[index]
        data = self.preprocess(raw_sample.copy())
        return data

    def __len__(self) -> int:
        """Get the number of samples in the dataset."""
        return len(self.raw_data)


class SupervisedTokenizedDataset(Dataset):

    def __init__(
        self,
        path: str,
        template: str | None = None,
        tokenizer: transformers.PreTrainedTokenizer | None = None,
        processor: transformers.ProcessorMixin | transforms.Compose | None = None,
        padding_side: str = 'right',
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
        self.padding_side = padding_side

        self.raw_data = torch.load(f'{path}/{data_files}', map_location=torch.device('cpu'))
        if size:
            self.raw_data = self.raw_data.select(range(int(size)))
        self.template = template

    def get_collator(self) -> Callable[[list[dict[str, torch.Tensor]]], dict[str, torch.Tensor]]:
        return SupervisedCollator(self.tokenizer.pad_token_id, self.processor, self.padding_side)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        """Get a tokenized data sample by index."""
        raw_sample = self.raw_data[index]
        return raw_sample

    def __len__(self) -> int:
        """Get the number of samples in the dataset."""
        return len(self.raw_data)


class SupervisedCollator:

    def __init__(self, pad_token_id: int, processor: transformers.ProcessorMixin | transforms.Compose | None = None, padding_side: str = 'right') -> None:
        self.pad_token_id = pad_token_id
        self.processor = processor
        self.padding_side = padding_side

    def __call__(self, samples: list[SupervisedSample]) -> SupervisedBatch:

        return_dict = {}
        current_device = get_current_device()

        return_dict['input_ids'] = right_padding(
            [sample['input_ids'] for sample in samples],
            padding_value=self.pad_token_id,
        ).to(current_device)

        return_dict['labels'] = right_padding(
            [sample['labels'] for sample in samples],
            padding_value=IGNORE_INDEX,
        ).to(current_device)

        if 'attention_mask' in samples[0]:
            return_dict['attention_mask'] = right_padding(
                [sample['attention_mask'] for sample in samples],
                padding_value=0,
            ).to(current_device)

        if 'pixel_values' in samples[0]:
            return_dict['pixel_values'] = right_padding(
                [sample['pixel_values'] for sample in samples],
                padding_value=0,
            ).to(current_device)

        if 'task' in samples[0]:
            return_dict['task'] = samples[0]['task']

        if "images_seq_mask" in samples[0]:
            return_dict['images_seq_mask'] = right_padding(
                [sample['images_seq_mask'] for sample in samples],
                padding_value=0,
            ).to(current_device)
            
        if "images_emb_mask" in samples[0]:
            return_dict['images_emb_mask'] = right_padding(
                [sample['images_emb_mask'] for sample in samples],
                padding_value=0,
            ).to(current_device)

        return return_dict
