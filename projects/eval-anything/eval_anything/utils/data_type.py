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

"""
数据类型定义
InferenceInput: 存储经过dataloader处理后的数据
EvaluationResult: 存储评估结果

TODO 还需适配
"""

import json
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Union

import numpy as np
import PIL
import torch
from openai.types.chat.chat_completion import ChatCompletion
from vllm.outputs import RequestOutput
from vllm.sequence import PromptLogprobs

from eval_anything.utils.uuid import UUIDGenerator


@dataclass
class RewardModelOutput:
    """The output data of a reward model."""


class ModalityType(Enum):
    TEXT = 'text'
    IMAGE = 'image'
    VIDEO = 'video'
    AUDIO = 'audio'
    VISION = 'vision'  # For Imagebind

    def __str__(self):
        return self.value

    def __eq__(self, other):
        if isinstance(other, ModalityType):
            return self.value == other.value
        elif isinstance(other, str):
            return self.value == other
        return False

    def __hash__(self):
        return hash(self.value)

    def is_valid_modality(cls, value: str) -> bool:
        try:
            cls(value)
            return True
        except ValueError:
            return False


class MultiModalData:
    def __init__(self, url: str | None, file):
        self.url = url
        self.file = file
        self.modality = self.get_modality()

    # TODO 从self.file获取modality
    def get_modality(self):
        if isinstance(self.file, PIL.Image.Image):
            return ModalityType.IMAGE
        elif isinstance(self.file, List[PIL.Image.Image]):
            return ModalityType.VIDEO
        elif isinstance(self.file, np.ndarray):
            return ModalityType.AUDIO
        else:
            raise ValueError(f'Unsupported file type: {type(self.file)}')


@dataclass
class InferenceInput:
    """The input data of a completion request to the LLM.

    Args:
        task: The task name.
        conversation: The conversation history.
        uuid: The unique id of the input.
        ref_answer: The ground truth answer.
        metadata: The metadata of the input.
    """

    task: str
    conversation: List[Dict]
    uuid: str
    ref_answer: str | dict[str, list] | List[any] | int | None
    metadata: dict

    def __init__(
        self,
        task: str,
        conversation: List[Dict[str, str]],
        ref_answer: str | dict[str, list] | List[any] | int | None = None,
        metadata: dict = None,
    ):
        self.task = task
        self.conversation = conversation
        self.ref_answer = ref_answer  # ground_truth
        self.metadata = metadata or {}  # Store benchmark-specific data

        self.uuid_generator = UUIDGenerator()
        self.uuid = self.uuid_generator({'task': self.task, 'conversation': self.conversation})

    def __str__(self):
        return json.dumps(
            {'task': self.task, 'conversation': self.conversation}, ensure_ascii=False, indent=4
        )

    def __repr__(self):
        return self.__str__()

    def __eq__(self, other):
        if not isinstance(other, InferenceInput):
            return False
        return self.uuid == other.uuid

    def to_dict(self):
        return {
            'task': self.task,
            'conversation': self.conversation,
            'uuid': self.uuid,
            'ref_answer': self.ref_answer,
        }


@dataclass
class InferenceOutput:
    """The output data of a completion request to the LLM.

    Args:
        task: The task name.
        uuid: The unique id of the input.
        engine: The inference engine used.
        response: The response string of the request.
        response_token_ids: The token IDs of the response string.
        response_logprobs: The logprobs of the response string.
        raw_output: The raw output data of the request.
        mm_data: The multi-modal data of the request.
    """

    task: str
    ref_answer: str | int
    uuid: str
    engine: str
    response: str
    response_token_ids: Optional[List[int]]
    response_logprobs: Optional[PromptLogprobs] | dict[str, list]
    raw_output: Optional[Union[RequestOutput, None]]
    mm_data: List[MultiModalData]  # TODO: left for mm-generation task
    label: str = None

    def __post_init__(self):
        pass

    def __init__(
        self,
        task: str,
        ref_answer: str | int,
        uuid: str,
        response: str,
        engine: str = 'hand',
        response_token_ids: Optional[List[int]] = None,
        response_logprobs: Optional[PromptLogprobs] | dict[str, list] = None,
        raw_output: Optional[Union[RequestOutput, None]] = None,
    ):
        self.engine = engine
        self.task = task
        self.ref_answer = ref_answer
        self.uuid = uuid
        self.response = response
        self.response_token_ids = response_token_ids
        self.response_logprobs = response_logprobs
        self.raw_output = raw_output

    @classmethod
    def from_vllm_output(
        cls, task, ref_answer, uuid, vllm_output: RequestOutput, store_raw: bool = False
    ):
        return cls(
            engine='vllm',
            task=task,
            ref_answer=ref_answer,
            uuid=uuid,
            response=[output.text for output in vllm_output.outputs],
            response_token_ids=[output.token_ids for output in vllm_output.outputs],
            response_logprobs=[output.logprobs for output in vllm_output.outputs],
            raw_output=vllm_output if store_raw else None,
        )

    @classmethod
    def from_hf_output(cls, task, uuid, response, hf_output: torch.Tensor, store_raw: bool = False):
        return cls(
            engine='huggingface',
            task=task,
            uuid=uuid,
            response=response,
            response_token_ids=hf_output.tolist(),
            raw_output=hf_output if store_raw else None,
        )

    def to_dict(self):
        return {
            'task': self.task,
            'uuid': self.uuid,
            'response': self.response,
            'engine': self.engine,
        }

    # TODO
    # @classmethod
    # def from_data(cls, data: Dict, store_raw: bool = False):
    # return cls(
    #     engine='dict',
    #     question_id=data.get('question_id'),
    #     prompt=data.get('prompt'),
    #     response=data.get('response'),
    #     raw_output=data if store_raw else None,
    # )

    # @classmethod
    # def from_dict(cls, data: Dict, store_raw: bool = False):
    #     return cls(
    #         engine='dict',
    #         prompt=data.get('prompt'),
    #         response=data.get('response'),
    #         question_id=data.get('question_id'),
    #         prompt_token_ids=data.get('prompt_token_ids'),
    #         prompt_logprobs=data.get('prompt_logprobs'),
    #         response_token_ids=data.get('response_token_ids'),
    #         response_logprobs=data.get('response_logprobs'),
    #         raw_output=data if store_raw else None,
    #     )

    # @classmethod
    # def from_deepspeed_output(cls, deepspeed_output: Dict, store_raw: bool = False):
    #     return cls(
    #         engine='deepspeed',
    #         prompt=deepspeed_output.get('prompt'),
    #         question_id=deepspeed_output.get('question_id'),
    #         prompt_token_ids=deepspeed_output.get('prompt_token_ids'),
    #         prompt_logprobs=deepspeed_output.get('prompt_logprobs'),
    #         response=deepspeed_output.get('response'),
    #         response_token_ids=deepspeed_output.get('response_token_ids'),
    #         response_logprobs=deepspeed_output.get('response_logprobs'),
    #         raw_output=deepspeed_output if store_raw else None,
    #     )

    # def __repr__(self):
    #     return (
    #         f'InferenceOutput('
    #         f'engine={self.engine!r}, '
    #         f'prompt={self.prompt!r}, '
    #         f'question_id={self.question_id!r}, '
    #         f'prompt_token_ids={self.prompt_token_ids!r}, '
    #         f'prompt_logprobs={self.prompt_logprobs!r}, '
    #         f'response={self.response!r}, '
    #         f'response_token_ids={self.response_token_ids!r}, '
    #         f'response_logprobs={self.response_logprobs!r})'
    #     )

    def __repr__(self):
        return (
            f'InferenceOutput('
            f'task={self.task!r}, '
            f'engine={self.engine!r}, '
            f'uuid={self.uuid!r}, '
            f'ref_answer={self.ref_answer!r}, '
            f'response={self.response!r}, '
            f'response_token_ids={self.response_token_ids!r}, '
            f'response_logprobs={self.response_logprobs!r})'
            f'raw_output={self.raw_output!r}'
        )


@dataclass
class EvaluationResult:

    def __init__(
        self,
        benchmark_name: str,
        inference_output: InferenceOutput,
        extracted_result: dict[str, any] | None,
        ground_truth: str | dict[str, list] | None,
        uuid: str,
    ):
        self.benchmark_name = benchmark_name
        self.inference_output = inference_output
        self.extracted_result = extracted_result
        self.ground_truth = ground_truth
        self.uuid = uuid
        self.evaluation_results = {}

    def to_dict(self) -> dict:
        return {
            'benchmark_name': self.benchmark_name,
            'inference_output': self.inference_output.to_dict(),
            'extracted_result': self.extracted_result,
            'ground_truth': self.ground_truth,
            'uuid': self.uuid,
        }

    def update_evaluation_result(self, metric_name: str, metric_result: float):
        self.evaluation_results[metric_name] = metric_result


@dataclass
class SingleInput:
    '''
    Args:
        prompt: The prompt string of the request.
        response: The response string of the request.
    '''

    prompt: str
    response: str

    def __init__(self, prompt: str, response: str):
        self.prompt = prompt
        self.response = response

    @classmethod
    def from_InferenceOutput(cls, inference: InferenceOutput):
        return cls(
            prompt=inference.prompt,
            response=inference.response,
        )

    def build_gpt_input(self, judge_prompt: str, template_function=None):
        prompt = template_function(self)
        return [{'role': 'system', 'content': judge_prompt}, {'role': 'user', 'content': prompt}]

    def __repr__(self):
        return f'SingleInput(prompt={self.prompt!r}, response={self.response!r})'


'''
Reward model: [InferenceOutput] -> [EvalOutput]
Arena GPT eval: [ArenaInput] -> [EvalOutput]

'''


def function1(ArenaInput):
    return 'Human: {prompt}\nAssistant 1: {response1}\nAssistant 2: {response2}'.format(
        prompt=ArenaInput.prompt, response1=ArenaInput.response1, response2=ArenaInput.response2
    )


MMdata = Dict[str, any]


@dataclass
class ArenaInput:
    """The input data of a pairwise evaluation request.

    Args:
        request_id: The unique ID of the request.
        prompt: "Human:....Assistant:..."
        response1: The first response string of the request.
        response2: The second response string of the request.
    """

    engine: str
    prompt: Union[str, MMdata]
    response1: Union[str, MMdata]
    response2: Union[str, MMdata]

    def __init__(
        self,
        prompt: Union[str, MMdata],
        response1: Union[str, MMdata],
        response2: Union[str, MMdata],
        engine: str = 'hand',
    ):
        self.engine = engine
        self.prompt = prompt
        self.response1 = response1
        self.response2 = response2

    @classmethod
    def from_InferenceOutput(cls, inference1: InferenceOutput, inference2: InferenceOutput):
        assert inference1.prompt == inference2.prompt
        return cls(
            engine='from_InferenceOutput',
            prompt=inference1.prompt,
            response1=inference1.response,
            response2=inference2.response,
        )

    def build_gpt_input(self, judge_prompt: str, template_function=None):
        prompt = template_function(self)
        return [{'role': 'system', 'content': judge_prompt}, {'role': 'user', 'content': prompt}]

    def __repr__(self) -> str:
        return (
            f'ArenaInput('
            f'engine={self.engine!r}, '
            f'prompt={self.prompt!r}, '
            f'response1={self.response1!r}, '
            f'response2={self.response2!r})'
        )


@dataclass
class EvalOutput:
    """
    Args:
        evalEngine: The evaluation engine used. \\
        input: The input data of the evaluation request. \\
        raw_output: The raw output data of the evaluation request.
         - For GPT evaluation, it is a ChatCompletion object. \\
         - For vllm evaluation, it is a RequestOutput object.
         - For something wrong, it is an Exception object.
    """

    evalEngine: str
    input: Union[SingleInput, ArenaInput]
    raw_output: Union[ChatCompletion, Exception, RequestOutput]

    def __post_init__(self):
        assert self.evalEngine in ['gpt_evaluation', 'arena']

    @classmethod
    def from_dict(cls, data: Dict):
        return cls(
            evalEngine=data.get('evalEngine'),
            input=data.get('input'),
            raw_output=data.get('raw_output'),
        )

    def parse_text(self):
        if isinstance(self.raw_output, ChatCompletion):
            return self.raw_output.choices[0].text
        elif isinstance(self.raw_output, RequestOutput):
            return self.raw_output.outputs[0].text
        else:
            return None

    def __repr__(self) -> str:
        return (
            f'EvalOutput('
            f'evalEngine={self.evalEngine!r}, '
            f'input={self.input!r}, '
            f'raw_output={self.raw_output!r})'
        )
