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


import os
from typing import Any, Callable
from typing_extensions import TypedDict  # Python 3.10+

import torch
import transformers
from torch.utils.data import Dataset
from torchvision import transforms
from transformers.tokenization_utils import PaddingStrategy, TruncationStrategy

from align_anything.utils.multi_process import get_current_device
from align_anything.utils.tools import convert_to_rgb, ends_with_any
from datasets import load_dataset


IGNORE_INDEX = -100

__all__ = [
    'SupervisedDataset',
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
        padding_side: str = 'right',
        name: str | None = None,
        size: int | None = None,
        split: str | None = None,
        data_files: str | None = None,
        optional_args: list | str = [],
    ):
        super().__init__()
        assert path, f'You must set the valid datasets path! Here is {path}'
        assert template, f'You must set the valid template path! Here is {template}'
        self.path = path
        self.tokenizer = tokenizer
        self.processor = processor
        self.padding_side = padding_side
        self.raw_data = load_dataset(
            path,
            name=name if name and name != 'None' else None,
            split=split if split and split != 'None' else None,
            data_files=data_files if data_files and data_files != 'None' else None,
            *optional_args,
            trust_remote_code=True,
            num_proc=16,
        )
        if size:
            self.raw_data = self.raw_data.select(range(int(size)))
        self.template = template

    def preprocess(self, raw_sample: dict[str, Any]) -> SupervisedSample:
        return_dict = {}
        prompt, conversation, meta_info = self.template.format_supervised_sample(raw_sample)
        if not ends_with_any(conversation, self.tokenizer.eos_token):
            conversation += self.tokenizer.eos_token

        # return necessary information
        return_dict['prompt'] = prompt
        return_dict['conversation'] = conversation
        return_dict['image'] = meta_info['image']

        # set the labels masked by the prompt
        return_dict['prompt_lens'] = len(
            self.tokenize(prompt, add_special_tokens=False)['input_ids'][0]
        )

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


class SupervisedCollator:

    def __init__(
        self,
        pad_token_id: int,
        processor: transformers.ProcessorMixin | transforms.Compose | None = None,
        padding_side: str = 'right',
    ) -> None:
        """Initialize a collator."""
        self.pad_token_id = pad_token_id
        self.processor = processor
        self.padding_side = padding_side

    def __call__(self, samples: list[SupervisedSample]) -> SupervisedBatch:
        return_dict = {'meta_info': {}}
        current_device = get_current_device()

        concated_text = [sample['conversation'] for sample in samples]

        if os.environ.get('MULTI_IMAGES_INFERENCE_MODELS') == 'Yes':
            images = [[sample['image']] for sample in samples]
        else:
            images = [sample['image'] for sample in samples]

        # FIXME: special for gemma3 processor, will be merge in next version
        if isinstance(self.processor, transformers.Gemma3Processor):
            images = [[convert_to_rgb(img)] for img in images]
            return_dict['meta_info']['images'] = images
        else:
            return_dict['meta_info']['images'] = images

        multi_modal_padding = self.processor(
            images=images,
            text=concated_text,
            return_tensors='pt',
            padding=True,
            padding_side=self.padding_side,
            return_attention_mask=True,
        )

        inputs_ids = multi_modal_padding['input_ids']
        labels = inputs_ids.clone()

        for i in range(len(samples)):
            prompt_lens = samples[i]['prompt_lens']
            labels[i, :prompt_lens] = IGNORE_INDEX

        return_dict.update(multi_modal_padding)
        return_dict['labels'] = labels
        for key, value in return_dict.items():
            if isinstance(value, torch.Tensor):
                return_dict[key] = value.to(current_device)
            elif key == 'pixel_values':

                def move_to_device(item):
                    if isinstance(item, list):
                        return [move_to_device(sub_item) for sub_item in item]
                    elif isinstance(item, torch.Tensor):
                        return item.to(current_device)
                    return item

                return_dict[key] = move_to_device(value)

        return return_dict
