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
from align_anything.utils.template_registry import get_template_class
from align_anything.utils.tools import left_padding
from datasets import load_dataset


IGNORE_INDEX = -100

__all__ = [
    'PromptOnlyDataset',
    'PromptOnlyCollator',
    'PromptOnlySample',
    'PromptOnlyBatch',
]


def remove_duplicate_prompts(dict_list: list[dict[str, Any]], template):
    seen_prompts = set()
    unique_dict_list = []
    for idx in range(len(dict_list)):
        item = dict_list[idx]
        prompt = template.format_prompt_only_sample(item)['text']
        if prompt not in seen_prompts:
            unique_dict_list.append(item)
            seen_prompts.add(prompt)
    return unique_dict_list


class PromptOnlySample(TypedDict, total=True):
    input_ids: torch.LongTensor  # size = (L,)
    labels: torch.LongTensor  # size = (L,)
    pixel_values: torch.LongTensor | None  # size = (B, C, H, W)


class PromptOnlyBatch(TypedDict, total=True):
    input_ids: torch.LongTensor  # size = (B, L)
    labels: torch.LongTensor  # size = (B, L)
    attention_mask: torch.BoolTensor  # size = (B, L)
    pixel_values: torch.LongTensor | None  # size = (B, C, H, W)
    image_sizes: list[int] | None


class PromptOnlyDataset(Dataset):

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
        raw_data_duplicated = load_dataset(
            path,
            split=split,
            data_files=data_files,
            subset=subset,
            *optional_args,
            trust_remote_code=True,
        )
        self.template = get_template_class(template)
        self.raw_data = remove_duplicate_prompts(raw_data_duplicated, self.template)

        if size:
            self.raw_data = self.raw_data[:int(size)]

    def preprocess(self, raw_sample: dict[str, Any]) -> PromptOnlySample:
        formatted_sample = self.template.format_prompt_only_sample(raw_sample)
        return_dict = {}
        raw_text = ''
        if isinstance(formatted_sample['text'], list):
            raw_text = self.tokenizer.eos_token.join(formatted_sample['text'])
        elif isinstance(formatted_sample['text'], str):
            raw_text = formatted_sample['text'] + self.tokenizer.eos_token
        else:
            raise NotImplementedError
        return_dict['input_ids'] = self.tokenize(raw_text)

        raw_image = formatted_sample['image']
        return_dict['pixel_values'] = self.processor.image_processor(
            raw_image, return_tensors='pt'
        )['pixel_values'][0]

        return return_dict

    def get_collator(self) -> Callable[[list[dict[str, torch.Tensor]]], dict[str, torch.Tensor]]:
        return PromptOnlyCollator(self.tokenizer.pad_token_id)

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
        data = self.preprocess(raw_sample)
        return data

    def __len__(self) -> int:
        """Get the number of samples in the dataset."""
        return len(self.raw_data)


class PromptOnlyCollator:

    def __init__(self, pad_token_id: int) -> None:
        """Initialize a collator."""
        self.pad_token_id = pad_token_id

    def __call__(self, samples: list[PromptOnlySample]) -> PromptOnlyBatch:
        return_dict = {}
        current_device = get_current_device()

        input_ids = [sample['input_ids'] for sample in samples]
        attention_mask = [
            input_id.new_ones(input_id.size(), dtype=torch.bool) for input_id in input_ids
        ]

        return_dict['input_ids'] = left_padding(input_ids, padding_value=self.pad_token_id).to(
            current_device
        )
        return_dict['attention_mask'] = left_padding(attention_mask, padding_value=0).to(
            current_device
        )


        if samples[0]['pixel_values'].dim() == 4:
            # init list for pixel_values

            _pixel_values_list = []
            for sample in samples:
                pixel_values = sample['pixel_values']  # size = (P, C, H, W)
                _pixel_values_list.append(pixel_values)

            return_dict['pixel_values'] = torch.cat(_pixel_values_list, dim=0).to(
                current_device
            )
            # size = (P1+P2+...+P_n+P1+P2+...+P_n, C, H, W)

            # image_sizes
            b = samples[0]['pixel_values'].shape[2]
            c = samples[0]['pixel_values'].shape[3]
            return_dict['image_sizes'] = [
                sample['pixel_values'].to(current_device).size(0) for sample in samples
            ]

        else:
            # original code for non-patches
            return_dict['pixel_values'] = torch.stack(
                [sample['pixel_values'] for sample in samples]
            ).to(current_device)

        return return_dict
