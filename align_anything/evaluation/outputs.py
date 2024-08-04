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

from typing import List, Optional, Union, Dict
from dataclasses import dataclass
from vllm.outputs import CompletionOutput, RequestOutput
from vllm.sequence import PromptLogprobs
from openai.types.chat.chat_completion import ChatCompletion

@dataclass
class RewardModelOutput:
    """The output data of a reward model.
    """

@dataclass
class InferenceInput:
    '''
    Args:
        text: The text to be completed.
            
    '''
    text: str
    image_url: Optional[str] = None

    def __init__(self, text: str):
        self.text = text

    def __repr__(self):
        return f"InferenceInput(text={self.text!r})"

@dataclass
class InferenceOutput:
    """The output data of a completion request to the LLM.

    Args:
        engine: The inference engine used. \\
        prompt: The prompt string of the request. \\
        prompt_token_ids: The token IDs of the prompt string. \\
        prompt_logprobs: The logprobs of the prompt string. \\
        response: The response string of the request. \\
        response_token_ids: The token IDs of the response string. \\
        response_logprobs: The logprobs of the response string.
    """

    engine: str
    prompt: str
    response: str
    prompt_token_ids: Optional[List[int]]
    prompt_logprobs: Optional[PromptLogprobs]
    response_token_ids: Optional[List[int]]
    response_logprobs: Optional[PromptLogprobs]

    def __post_init__(self):
        pass
    
    def __init__(self, 
                 prompt: str, 
                 response: str,
                 engine: str = "hand",
                 prompt_token_ids: Optional[List[int]] = None,
                 prompt_logprobs: Optional[PromptLogprobs] = None,
                 response_token_ids: Optional[List[int]] = None,
                 response_logprobs: Optional[PromptLogprobs] = None
                ):
        self.engine = engine
        self.prompt = prompt
        self.prompt_token_ids = prompt_token_ids
        self.prompt_logprobs = prompt_logprobs
        self.response = response
        self.response_token_ids = response_token_ids
        self.response_logprobs = response_logprobs

    @classmethod
    def from_vllm_output(cls, vllm_output: RequestOutput):
        return cls(
            engine="vllm",
            prompt=vllm_output.prompt,
            prompt_token_ids=vllm_output.prompt_token_ids,
            prompt_logprobs=vllm_output.prompt_logprobs,
            response=vllm_output.outputs[0].text,
            response_token_ids=vllm_output.outputs[0].token_ids,
            response_logprobs=vllm_output.outputs[0].logprobs
        )

    @classmethod
    def from_dict(cls, data: Dict):
        return cls(
            engine="dict",
            prompt=data.get("prompt"),
            response=data.get("response"),
            prompt_token_ids=data.get("prompt_token_ids"),
            prompt_logprobs=data.get("prompt_logprobs"),
            response_token_ids=data.get("response_token_ids"),
            response_logprobs=data.get("response_logprobs")
        )
    
    def from_deepspeed_output(self, deepspeed_output: Dict):
        pass

    def __repr__(self):
        return (
            f"InferenceOutput("
            f"engine={self.engine!r}, "
            f"prompt={self.prompt!r}, "
            f"prompt_token_ids={self.prompt_token_ids!r}, "
            f"prompt_logprobs={self.prompt_logprobs!r}, "
            f"response={self.response!r}, "
            f"response_token_ids={self.response_token_ids!r}, "
            f"response_logprobs={self.response_logprobs!r})"
        )

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
        return [
            {'role': 'system', 'content': judge_prompt},
            {'role': 'user', 'content': prompt}
        ]

    def __repr__(self):
        return f"SingleInput(prompt={self.prompt!r}, response={self.response!r})"

'''
Reward model: [InferenceOutput] -> [EvalOutput]
Arena GPT eval: [ArenaInput] -> [EvalOutput]

'''

def function1(ArenaInput):
    return "Human: {prompt}\nAssistant 1: {response1}\nAssistant 2: {response2}".format(prompt=ArenaInput.prompt, response1=ArenaInput.response1, response2=ArenaInput.response2)

MMdata = Dict[str,any]
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

    def __init__(self, 
                 prompt: Union[str, MMdata], 
                 response1: Union[str, MMdata], 
                 response2: Union[str, MMdata],
                 engine: str = "hand",
                ):
        self.engine = engine
        self.prompt = prompt
        self.response1 = response1
        self.response2 = response2

    @classmethod
    def from_InferenceOutput(cls, inference1: InferenceOutput, inference2: InferenceOutput):
        assert inference1.prompt == inference2.prompt
        return cls(
            engine="from_InferenceOutput",
            prompt=inference1.prompt,
            response1=inference1.response,
            response2=inference2.response,
            
        )

    def build_gpt_input(self, judge_prompt: str, template_function=None):
        prompt = template_function(self)
        return [
            {'role': 'system', 'content': judge_prompt},
            {'role': 'user', 'content': prompt}
        ]

    def __repr__(self) -> str:
        return (f"ArenaInput("
                f"engine={self.engine!r}, "
                f"prompt={self.prompt!r}, "
                f"response1={self.response1!r}, "
                f"response2={self.response2!r})")


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
    input : Union[SingleInput, ArenaInput]
    raw_output: Union[ChatCompletion, Exception, RequestOutput]

    def __post_init__(self):
        assert self.evalEngine in ["gpt_evaluation", "arena"]

    @classmethod
    def from_dict(cls, data: Dict):
        return cls(
            evalEngine=data.get("evalEngine"),
            input=data.get("input"),
            raw_output=data.get("raw_output")
        )

    def parse_text(self):
        if isinstance(self.raw_output, ChatCompletion):
            return self.raw_output.choices[0].text
        elif isinstance(self.raw_output, RequestOutput):
            return self.raw_output.outputs[0].text
        else:
            return None

    def __repr__(self) -> str:
        return (f"EvalOutput("
                f"evalEngine={self.evalEngine!r}, "
                f"input={self.input!r}, "
                f"raw_output={self.raw_output!r})")