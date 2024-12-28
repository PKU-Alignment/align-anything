# Copyright 2024 PKU-Alignment Team and Lagent Team. All Rights Reserved.
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

from typing import List, Union

from agent.llms.base_llm import BaseModel
from agent.utils.util import filter_suffix


class VllmModel(BaseModel):
    """
    A wrapper of vLLM model.

    Args:
        path (str): The path to the model.
            It could be one of the following options:
                    - i) A local directory path of a huggingface model.
                    - ii) The model_id of a model hosted inside a model repo
                        on huggingface.co, such as "internlm/internlm-chat-7b",
                        "Qwen/Qwen-7B-Chat ", "baichuan-inc/Baichuan2-7B-Chat"
                        and so on.
        tp (int): tensor parallel
        vllm_cfg (dict): Other kwargs for vllm model initialization.
    """

    def __init__(self, path: str, tp: int = 1, vllm_cfg=dict(), **kwargs):

        super().__init__(path=path, **kwargs)
        from vllm import LLM

        self.model = LLM(
            model=self.path, trust_remote_code=True, tensor_parallel_size=tp, **vllm_cfg
        )

    def generate(
        self,
        inputs: Union[str, List[str]],
        do_preprocess: bool = None,
        skip_special_tokens: bool = False,
        **kwargs,
    ):
        """Return the chat completions in non-stream mode.

        Args:
            inputs (Union[str, List[str]]): input texts to be completed.
            do_preprocess (bool): whether pre-process the messages. Default to
                True, which means chat_template will be applied.
            skip_special_tokens (bool): Whether or not to remove special tokens
                in the decoding. Default to be False.
        Returns:
            (a list of/batched) text/chat completion
        """
        from vllm import SamplingParams

        batched = True
        if isinstance(inputs, str):
            inputs = [inputs]
            batched = False
        prompt = inputs
        gen_params = self.update_gen_params(**kwargs)
        max_new_tokens = gen_params.pop('max_new_tokens')
        stop_words = gen_params.pop('stop_words')

        sampling_config = SamplingParams(
            skip_special_tokens=skip_special_tokens,
            max_tokens=max_new_tokens,
            stop=stop_words,
            **gen_params,
        )
        response = self.model.generate(prompt, sampling_params=sampling_config)
        response = [resp.outputs[0].text for resp in response]
        # remove stop_words
        response = filter_suffix(response, self.gen_params.get('stop_words'))
        if batched:
            return response
        return response[0]
