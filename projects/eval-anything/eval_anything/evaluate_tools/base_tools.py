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

from abc import ABC, abstractmethod

from eval_anything.utils.logger import EvalLogger


prompt_builder_logger = EvalLogger(name='PromptBuilderLogger')
metric_logger = EvalLogger(name='MetricLogger')
tool_logger = EvalLogger(name='ToolLogger')
mm_data_manager_logger = EvalLogger(name='MMDataManagerLogger')


class BaseTool(ABC):
    def __init__(self, **kwargs) -> None:
        for key, value in kwargs.items():
            setattr(self, key, value)
        self.logger = tool_logger

    @abstractmethod
    def apply(self, **kwargs):
        pass

    def __call__(self, **kwargs):
        return self.apply(**kwargs)


class BaseMetric(ABC):
    def __init__(self, **kwargs) -> None:
        for key, value in kwargs.items():
            setattr(self, key, value)
        self.logger = metric_logger

    @abstractmethod
    def calculate(self):
        pass

    def __call__(self, **kwargs):
        return self.calculate(**kwargs)


class PromptBuilder(ABC):
    def __init__(self, **kwargs) -> None:
        for key, value in kwargs.items():
            setattr(self, key, value)
        self.logger = prompt_builder_logger

    @abstractmethod
    def build_prompt(self, **kwargs):
        pass


class BaseMMDataManager(ABC):
    def __init__(self, **kwargs) -> None:
        for key, value in kwargs.items():
            setattr(self, key, value)
        self.logger = mm_data_manager_logger

    @abstractmethod
    def extract_from_conversation(self, conversation):
        pass

    @abstractmethod
    def prompt_to_conversation(self, user_prompt: str, system_prompt: str, **kwargs):
        pass

    @abstractmethod
    def encode_to_base64(self, data, **kwargs):
        pass

    @abstractmethod
    def decode_from_base64(self, base64_string: str, **kwargs):
        pass
