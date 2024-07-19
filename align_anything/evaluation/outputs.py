'''
# This file uses code from the vllm library.
# vllm: https://github.com/vllm-project/vllm
# License: Apache-2.0 license

'''

from typing import List, Optional, Union, Dict
from dataclasses import dataclass

from vllm.outputs import CompletionOutput, RequestOutput
from vllm.sequence import PromptLogprobs

@dataclass
class InferenceOutput:
    """The output data of a completion request to the LLM.

    Args:
        request_id: The unique ID of the request.
        prompt: The prompt string of the request.
        prompt_token_ids: The token IDs of the prompt.
        prompt_logprobs: The log probabilities to return per prompt token.
        outputs: The output sequences of the request.
    """

    engine: str
    prompt: str
    prompt_token_ids: Optional[List[int]]
    prompt_logprobs: Optional[PromptLogprobs]
    response: str
    response_token_ids: Optional[List[int]]
    response_logprobs: Optional[PromptLogprobs]

    def __post_init__(self):
        assert self.engine in ["deepspeed", "vllm", "dict"]

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
            prompt_token_ids=data.get("prompt_token_ids"),
            prompt_logprobs=data.get("prompt_logprobs"),
            response=data.get("response"),
            response_token_ids=data.get("response_token_ids"),
            response_logprobs=data.get("response_logprobs")
        )
    
    def from_deepspeed_output(self, deepspeed_output: Dict):
        # todo
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
from dataclasses import dataclass

'''
Reward model: [InferenceOutput] -> [EvalOutput]
Arena GPT eval: [Arena_input] -> [EvalOutput]

'''

MMdata = Dict[str,str] # MultiModal data,like {'text':'','image_url':''}
@dataclass
class Arena_input:
    """The input data of a pairwise evaluation request.

    Args:
        request_id: The unique ID of the request.
        prompt: "Human:....Assistant:..."
        response1: The first response string of the request.
        response2: The second response string of the request.
        template: The template string of the request.
    """

    engine: str
    prompt: Union[str, MMdata]
    response1: Union[str, MMdata]
    response2: Union[str, MMdata]

    @classmethod
    def from_InferenceOutput(cls, inference1: InferenceOutput, inference2: InferenceOutput):
        assert inference1.prompt == inference2.prompt
        return cls(
            engine="from_InferenceOutput",
            prompt=inference1.prompt,
            response1=inference1.response,
            response2=inference2.response,
        )

    def __repr__(self) -> str:
        return (f"Arena_input("
                f"engine={self.engine!r}, "
                f"prompt={self.prompt!r}, "
                f"response1={self.response1!r}, "
                f"response2={self.response2!r})")


@dataclass
class EvalOutput:
    """
    Args:
        engine: The evaluation engine used
    """

    evalEngine: str
    input : Union[InferenceOutput, Arena_input]
    raw_output: Dict

    def __post_init__(self):
        assert self.evalEngine in ["gpt_evaluation", "arena"]

    @classmethod
    def from_dict(cls, data: Dict):
        return cls(
            evalEngine=data.get("evalEngine"),
            input=data.get("input"),
            raw_output=data.get("raw_output")
        )

    def __repr__(self) -> str:
        return (f"EvalOutput("
                f"evalEngine={self.evalEngine!r}, "
                f"input={self.input!r}, "
                f"raw_output={self.raw_output!r})")
