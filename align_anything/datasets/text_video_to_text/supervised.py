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
from transformers import LlavaNextVideoProcessor, Qwen2VLProcessor
from transformers.tokenization_utils import PaddingStrategy, TruncationStrategy

from align_anything.utils.multi_process import get_current_device, print_on_main_process
from align_anything.utils.process_llava_next_video import read_video_pyav as llava_next_video_loader
from align_anything.utils.process_qwen2vl import process_video_info as qwen2vl_video_loader
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
        )
        if size:
            self.raw_data = self.raw_data.select(range(int(size)))
        self.template = template

    def preprocess(self, raw_sample: dict[str, Any]) -> SupervisedSample:
        return_dict = {}
        prompt, conversation, meta_info = self.template.format_supervised_sample(raw_sample)
        if self.tokenizer.eos_token not in conversation:
            conversation += self.tokenizer.eos_token

        # return necessary information
        return_dict['prompt'] = prompt
        return_dict['conversation'] = conversation
        return_dict['video_info'] = {'video': meta_info['video']}

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
        data = self.preprocess(raw_sample)
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
        self.processor.tokenizer.padding_side = padding_side
        if isinstance(self.processor, Qwen2VLProcessor):
            self.video_loader = qwen2vl_video_loader
        elif isinstance(self.processor, LlavaNextVideoProcessor):
            self.video_loader = llava_next_video_loader
        else:
            self.video_loader = llava_next_video_loader
            print_on_main_process(
                """Using pre-processing method in
                  align_anything/utils/process_llava_next_video.py as the default video loader,
                  If you want to use other video pre-processing methods,
                  please modify the code in
                  align_anything/datasets/text_video_to_text/supervised.py"""
            )

    def __call__(self, samples: list[SupervisedSample]) -> SupervisedBatch:
        return_dict = {}
        current_device = get_current_device()

        concated_text = [sample['conversation'] for sample in samples]
        videos = [self.video_loader(sample['video_info']) for sample in samples]

        multi_modal_padding = self.processor(
            videos=videos,
            text=concated_text,
            return_tensors='pt',
            padding=True,
        )

        inputs_ids = multi_modal_padding['input_ids']
        labels = inputs_ids.clone()

        for i in range(len(samples)):
            prompt_lens = samples[i]['prompt_lens']
            labels[i, :prompt_lens] = IGNORE_INDEX

        return_dict['labels'] = labels.to(current_device)

        for key, value in multi_modal_padding.items():
            if isinstance(value, torch.Tensor):
                return_dict[key] = value.to(current_device)

        return return_dict
