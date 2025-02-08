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

from typing import Any, Callable
from typing_extensions import TypedDict  # Python 3.10+

import torch
import transformers
from torch.utils.data import Dataset
from torchvision import transforms
from transformers.tokenization_utils import PaddingStrategy, TruncationStrategy

from align_anything.utils.multi_process import get_current_device
from align_anything.utils.tools import right_padding
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


class SupervisedDataset(Dataset):

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
        formatted_sample = self.template.format_sample(raw_sample)
        return_dict = {}
        if 'input_image' in formatted_sample and formatted_sample['input_image'] is not None:

            full_conversation = [
                {
                    'role': 'User',
                    'content': formatted_sample['input_text'],
                    'images': (
                        [formatted_sample['input_image']]
                        if isinstance(formatted_sample['input_image'], str)
                        else formatted_sample['input_image']
                    ),
                },
                {'role': 'Assistant', 'content': formatted_sample['output_text']},
            ]

            prompt_conversation = [
                {
                    'role': 'User',
                    'content': formatted_sample['input_text'],
                    'images': (
                        [formatted_sample['input_image']]
                        if isinstance(formatted_sample['input_image'], str)
                        else formatted_sample['input_image']
                    ),
                },
                {
                    'role': 'Assistant',
                    'content': '',
                },
            ]
            full_inputs = self.processor(
                full_conversation, formatted_sample['input_image'], return_tensors='pt'
            )
            prompt_inputs = self.processor(
                prompt_conversation, formatted_sample['input_image'], return_tensors='pt'
            )

            return_dict = full_inputs.copy()
            return_dict['labels'] = return_dict['input_ids'].clone()
            return_dict['labels'][: len(prompt_inputs['input_ids'])] = IGNORE_INDEX
        elif 'output_image' in formatted_sample and formatted_sample['output_image'] is not None:
            raise NotImplementedError(
                'Not implemented inside SupervisedDataset. Please follow the instructions in projects/janus/README.md to deal with image input.'
            )
        return return_dict

    def get_collator(self) -> Callable[[list[dict[str, torch.Tensor]]], dict[str, torch.Tensor]]:
        return SupervisedCollator(self.tokenizer, self.processor)

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


class SupervisedTokenizedDataset(Dataset):

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

        self.raw_data = torch.load(f"{path}/{data_files}", map_location=torch.device('cpu'))
        if size:
            self.raw_data = self.raw_data.select(range(int(size)))
        self.template = template

    def get_collator(self) -> Callable[[list[dict[str, torch.Tensor]]], dict[str, torch.Tensor]]:
        return SupervisedCollator(self.tokenizer.pad_token_id)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        """Get a tokenized data sample by index."""
        raw_sample = self.raw_data[index]
        return raw_sample

    def __len__(self) -> int:
        """Get the number of samples in the dataset."""
        return len(self.raw_data)


class SupervisedCollator:

    def __init__(self, pad_token_id: int) -> None:
        """Initialize a collator."""
        self.pad_token_id = pad_token_id

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

        if 'attention_mask' in return_dict:
            return_dict['attention_mask'] = right_padding(
                [sample['attention_mask'] for sample in samples],
                padding_value=0,
            ).to(current_device)

        if 'pixel_values' in return_dict:
            new_samples = []

            for sample in samples:

                sample['pixel_values'] = torch.cat(
                    [tensor.to(current_device) for tensor in sample['pixel_values']], dim=0
                )

                new_samples.append(sample)

            _pixel_values_list = []
            for sample in new_samples:
                pixel_values = sample['pixel_values']  # size = (P, C, H, W)
                _pixel_values_list.append(pixel_values)

            return_dict['pixel_values'] = torch.cat(_pixel_values_list, dim=0).to(current_device)

        return return_dict
