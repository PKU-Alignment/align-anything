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


from typing import Any, Callable
from transformers import AutoTokenizer, AutoProcessor

class ModelFormatter():

    def __init__(self, formatter: AutoTokenizer | AutoProcessor, custom_formatter: Callable | None = None) -> None:
        self.formatter: AutoTokenizer | AutoProcessor = formatter
        if custom_formatter:
            self.format_sample = custom_formatter
        elif hasattr(formatter, 'apply_chat_template') and formatter.chat_template:
            self.format_sample = self.format_with_template
        else:
            self.format_sample = self.default_format

    def __call__(self, raw_sample: list[dict[str, Any]], add_generation_prompt: bool = False) -> str:
        return self.format_sample(raw_sample, add_generation_prompt)

    def default_format(self, raw_sample: list[dict[str, Any]], add_generation_prompt: bool = False) -> str:
        final_text = ''
        for line in raw_sample:
            for content in line['content']:
                if content['type'] == 'text':
                    final_text += line['role'].upper() + ': ' + content['text'] + '\n'
        if add_generation_prompt:
            final_text += 'ASSISTANT: '
        return final_text

    def format_with_template(self, raw_sample: list[dict[str, Any]], add_generation_prompt: bool = False) -> str:
        return self.formatter.apply_chat_template(raw_sample, tokenize=False, add_generation_prompt=add_generation_prompt)
