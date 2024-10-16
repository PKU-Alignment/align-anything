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
from align_anything.utils.tools import right_padding
from datasets import load_dataset


__all__ = [
    'PreferenceDataset',
    'PreferenceTokenizedDataset',
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


class PreferenceDataset(Dataset):

    def __init__(
        self,
        path: str,
        template: str,
        tokenizer: transformers.PreTrainedTokenizer,
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
        self.template = get_template_class(template)

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
        self.valid_indices = self.filter_indices()

        if size:
            size = min(size, len(self.raw_data))
            self.raw_data = self.raw_data.select(range(int(size)))

    def filter_indices(self):
        valid_indices = []
        for i, item in enumerate(self.raw_data):
            if not self.template.check_equal(item):
                valid_indices.append(i)
        return valid_indices

    def preprocess(self, raw_sample: dict[str, Any]) -> PreferenceSample:
        formatted_sample = self.template.format_preference_sample(raw_sample)
        return_dict = {}

        raw_better_text = ''
        raw_worse_text = ''

        if isinstance(formatted_sample['better_text'], list):
            raw_better_text = self.tokenizer.eos_token.join(formatted_sample['better_text'])
            raw_worse_text = self.tokenizer.eos_token.join(formatted_sample['worse_text'])
        elif isinstance(formatted_sample['better_text'], str):
            raw_better_text = formatted_sample['better_text'] + self.tokenizer.eos_token
            raw_worse_text = formatted_sample['worse_text'] + self.tokenizer.eos_token
        else:
            raise NotImplementedError
        return_dict['better_input_ids'] = self.processor(raw_better_text, formatted_sample['better_images'], return_tensors='pt')
        return_dict['worse_input_ids'] = self.processor(raw_worse_text, formatted_sample['worse_images'], return_tensors='pt')

        return return_dict

    def get_collator(self) -> Callable[[list[dict[str, torch.Tensor]]], dict[str, torch.Tensor]]:
        return PreferenceCollator(self.tokenizer.pad_token_id)


    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        """Get a tokenized data sample by index."""
        raw_sample = self.raw_data[index]
        data = self.preprocess(raw_sample)
        return data

    def __len__(self) -> int:
        """Get the number of samples in the dataset."""
        return len(self.valid_indices)

class PreferenceTokenizedDataset(Dataset):

    def __init__(
        self,
        path: str,
        template: str,
        tokenizer: transformers.PreTrainedTokenizer,
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
        self.template = get_template_class(template)
        
        self.raw_data = torch.load(f"{path}/{data_files}", map_location=torch.device('cpu'))
        self.valid_indices = self.filter_indices()
        if size:
            self.raw_data = self.raw_data.select(range(int(size)))

    def filter_indices(self):
        valid_indices = []
        for i, item in enumerate(self.raw_data):
            if not self.template.check_equal(item):
                valid_indices.append(i)
        return valid_indices

    def get_collator(self) -> Callable[[list[dict[str, torch.Tensor]]], dict[str, torch.Tensor]]:
        return PreferenceCollator(self.tokenizer.pad_token_id)


    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        """Get a tokenized data sample by index."""
        raw_sample = self.raw_data[index]
        return raw_sample

    def __len__(self) -> int:
        """Get the number of samples in the dataset."""
        return len(self.valid_indices)
    
class PreferenceCollator:

    def __init__(self, pad_token_id: int) -> None:
        """Initialize a collator."""
        self.pad_token_id = pad_token_id

    def __call__(self, samples: list[PreferenceSample]) -> tuple[PreferenceBatch]:
        return_dict = {}
        current_device = get_current_device()

        input_ids = [sample['better_input_ids'] for sample in samples] + [
            sample['worse_input_ids'] for sample in samples
        ]  # size = (2 * B, L)
        return_dict['input_ids'] = right_padding(input_ids, padding_value=self.pad_token_id).to(
            current_device
        )  # size = (2 * B, L)

        attention_mask = [
            input_id.new_ones(input_id.size(), dtype=torch.bool) for input_id in input_ids
        ]  # size = (2 * B, L)
        return_dict['attention_mask'] = right_padding(attention_mask, padding_value=0).to(
            current_device
        )  # size = (2 * B, L)

        if 'pixel_values' in samples[0].keys():

            a = return_dict['attention_mask'].shape[0]

            if samples[0]['pixel_values'].dim() == 4:
                # init list for pixel_values
                ori_patches = [
                    sample['pixel_values'].to(current_device).size(0) for sample in samples
                ]
                ori_patches_tensor = torch.tensor(ori_patches)
                double_ori_patches_tensor = torch.cat(
                    [ori_patches_tensor, ori_patches_tensor], dim=0
                )
                return_dict['image_sizes'] = double_ori_patches_tensor.to(current_device)

                _pixel_values_list = []
                for sample in samples:
                    pixel_values = sample['pixel_values']  # size = (P, C, H, W)
                    _pixel_values_list.append(pixel_values)

                pixel_values_tensor = torch.cat(_pixel_values_list, dim=0).to(current_device)
                double_stacked = torch.cat([pixel_values_tensor, pixel_values_tensor], dim=0)
                return_dict['pixel_values'] = double_stacked.to(current_device)

                # size = (P1+P2+...+P_n+P1+P2+...+P_n, C, H, W)

            else:
                # original code for non-patches
                pixel_values_tensor = torch.stack(
                    [sample['pixel_values'] for sample in samples]
                ).to(current_device)
                double_stacked = torch.cat([pixel_values_tensor, pixel_values_tensor], dim=0)
                return_dict['pixel_values'] = double_stacked.to(current_device)

        return return_dict
