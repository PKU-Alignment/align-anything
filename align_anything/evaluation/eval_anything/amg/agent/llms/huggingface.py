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


import copy
import logging
from typing import Dict, List, Optional, Union

from agent.llms.base_api import APITemplateParser
from agent.llms.base_llm import BaseModel
from agent.schema import ModelStatusCode

from align_anything.utils.device_utils import (
    get_current_device,
)


logger = logging.getLogger(__name__)

logger = logging.getLogger(__name__)


class HFTransformer(BaseModel):
    """Model wrapper around HuggingFace general models.

    Args:
        path (str): The name or path to HuggingFace's model.
        tokenizer_path (str): The path to the tokenizer. Defaults to None.
        tokenizer_kwargs (dict): Keyword arguments for the tokenizer.
            Defaults to {}.
        tokenizer_only (bool): If True, only the tokenizer will be initialized.
            Defaults to False.
        model_kwargs (dict): Keyword arguments for the model, used in loader.
            Defaults to dict(device_map='auto').
        meta_template (Dict, optional): The model's meta prompt
            template if needed, in case the requirement of injecting or
            wrapping of any meta instructions.
    """

    def __init__(
        self,
        path: str,
        tokenizer_path: Optional[str] = None,
        tokenizer_kwargs: dict = dict(),
        tokenizer_only: bool = False,
        model_kwargs: dict = dict(device_map='auto'),
        meta_template: Optional[Dict] = None,
        stop_words_id: Union[List[int], int] = None,
        **kwargs,
    ):
        super().__init__(
            path=path, tokenizer_only=tokenizer_only, meta_template=meta_template, **kwargs
        )
        if isinstance(stop_words_id, int):
            stop_words_id = [stop_words_id]
        self.gen_params.update(stop_words_id=stop_words_id)
        if (
            self.gen_params['stop_words'] is not None
            and self.gen_params['stop_words_id'] is not None
        ):
            logger.warning(
                'Both stop_words and stop_words_id are specified,'
                'only stop_words_id will be used.'
            )

        self._load_tokenizer(
            path=path, tokenizer_path=tokenizer_path, tokenizer_kwargs=tokenizer_kwargs
        )
        if not tokenizer_only:
            self._load_model(path=path, model_kwargs=model_kwargs)

        from transformers.generation.utils import (  # noqa: E501
            LogitsProcessorList,
            StoppingCriteriaList,
        )

        self.logits_processor = LogitsProcessorList()
        self.stopping_criteria = StoppingCriteriaList()
        self.prefix_allowed_tokens_fn = None

        stop_words_id = []
        if self.gen_params.get('stop_words_id'):
            stop_words_id = self.gen_params.get('stop_words_id')
        elif self.gen_params.get('stop_words'):
            for sw in self.gen_params.get('stop_words'):
                stop_words_id.append(self.tokenizer(sw)['input_ids'][-1])
        self.additional_eos_token_id = stop_words_id

    def _load_tokenizer(self, path: str, tokenizer_path: Optional[str], tokenizer_kwargs: dict):
        from transformers import AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path if tokenizer_path else path, trust_remote_code=True, **tokenizer_kwargs
        )

        if self.tokenizer.pad_token_id is None:
            if self.tokenizer.eos_token is not None:
                logger.warning(f'Using eos_token {self.tokenizer.eos_token} ' 'as pad_token.')
                self.tokenizer.pad_token = self.tokenizer.eos_token
            else:
                from transformers.generation import GenerationConfig

                self.gcfg = GenerationConfig.from_pretrained(path)

                if self.gcfg.pad_token_id is not None:
                    logger.warning(
                        f'Using pad_token_id {self.gcfg.pad_token_id} ' 'as pad_token_id.'
                    )
                    self.tokenizer.pad_token_id = self.gcfg.pad_token_id
                else:
                    raise ValueError(
                        'pad_token_id is not set for this tokenizer. Try to '
                        'set pad_token_id via passing '
                        '`pad_token_id={PAD_TOKEN_ID}` in model_cfg.'
                    )

    def _load_model(self, path: str, model_kwargs: dict):
        import torch
        from transformers import AutoModelForCausalLM

        model_kwargs.setdefault('torch_dtype', torch.float16)
        self.model = AutoModelForCausalLM.from_pretrained(
            path, trust_remote_code=True, **model_kwargs
        )
        self.model.eval()

        self.model._prepare_special_tokens(self.model.generation_config)

    def tokenize(self, inputs: str):
        assert isinstance(inputs, str)
        inputs = self.tokenizer(inputs, return_tensors='pt', return_length=True)
        return inputs['input_ids'].tolist()

    def generate(
        self,
        inputs: Union[str, List[str]],
        do_sample: bool = True,
        **kwargs,
    ):
        """Return the chat completions in non-stream mode.

        Args:
            inputs (Union[str, List[str]]): input texts to be completed.
            do_sample (bool): do sampling if enabled
        Returns:
            (a list of/batched) text/chat completion
        """
        for status, chunk, output_token_id in self.stream_generate(inputs, do_sample, **kwargs):
            response = chunk
        return response, output_token_id

    def stream_generate(
        self,
        inputs: List[str],
        do_sample: bool = True,
        **kwargs,
    ):
        """Return the chat completions in stream mode.

        Args:
            inputs (Union[str, List[str]]): input texts to be completed.
            do_sample (bool): do sampling if enabled
        Returns:
            tuple(Status, str, int): status, text/chat completion,
            generated token number
        """
        import torch
        from torch import nn

        with torch.no_grad():
            batched = True
            if isinstance(inputs, str):
                inputs = [inputs]
                batched = False
            inputs = self.tokenizer(inputs, padding=True, return_tensors='pt', return_length=True)
            input_length = inputs['length']
            for k, v in inputs.items():
                inputs[k] = v.to(get_current_device())
            input_ids = inputs['input_ids']
            attention_mask = inputs['attention_mask']
            batch_size = input_ids.shape[0]
            input_ids_seq_length = input_ids.shape[-1]
            generation_config = self.model.generation_config
            generation_config = copy.deepcopy(generation_config)
            new_gen_params = self.update_gen_params(**kwargs)
            generation_config.update(**new_gen_params)
            generation_config.update(**kwargs)
            model_kwargs = generation_config.to_dict()
            model_kwargs['attention_mask'] = attention_mask
            model_kwargs = self.model._get_initial_cache_position(input_ids, model_kwargs)
            _, eos_token_id = (  # noqa: F841  # pylint: disable=W0612
                generation_config.bos_token_id,
                generation_config.eos_token_id,
            )
            if eos_token_id is None:
                if self.gcfg.eos_token_id is not None:
                    eos_token_id = self.gcfg.eos_token_id
                else:
                    eos_token_id = []
            if isinstance(eos_token_id, int):
                eos_token_id = [eos_token_id]
            if self.additional_eos_token_id is not None:
                eos_token_id.extend(self.additional_eos_token_id)
            eos_token_id_tensor = (
                torch.tensor(eos_token_id).to(input_ids.device)
                if eos_token_id is not None
                else None
            )
            generation_config.max_length = generation_config.max_new_tokens + input_ids_seq_length
            # Set generation parameters if not already defined
            logits_processor = self.logits_processor
            stopping_criteria = self.stopping_criteria

            logits_processor = self.model._get_logits_processor(
                generation_config=generation_config,
                input_ids_seq_length=input_ids_seq_length,
                encoder_input_ids=input_ids,
                prefix_allowed_tokens_fn=self.prefix_allowed_tokens_fn,
                logits_processor=logits_processor,
            )

            stopping_criteria = self.model._get_stopping_criteria(
                generation_config=generation_config, stopping_criteria=stopping_criteria
            )
            # logits_warper = self.model._get_logits_warper(generation_config)

            unfinished_sequences = input_ids.new(batch_size).fill_(1)
            scores = None
            while True:
                model_inputs = self.model.prepare_inputs_for_generation(input_ids, **model_kwargs)
                # forward pass to get next token
                outputs = self.model(
                    **model_inputs,
                    return_dict=True,
                    output_attentions=False,
                    output_hidden_states=False,
                )

                next_token_logits = outputs.logits[:, -1, :]

                # pre-process distribution
                next_token_scores = logits_processor(input_ids, next_token_logits)
                # next_token_scores = logits_warper(input_ids, next_token_scores)

                # sample
                probs = nn.functional.softmax(next_token_scores, dim=-1)
                if do_sample:
                    next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
                else:
                    next_tokens = torch.argmax(probs, dim=-1)

                # update generated ids, model inputs,
                # and length for next step
                input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
                model_kwargs = self.model._update_model_kwargs_for_generation(  # noqa: E501
                    outputs, model_kwargs, is_encoder_decoder=False
                )
                unfinished_sequences = unfinished_sequences.mul(
                    next_tokens.tile(eos_token_id_tensor.shape[0], 1)
                    .ne(eos_token_id_tensor.unsqueeze(1))
                    .prod(dim=0)
                )
                output_token_ids = input_ids.cpu().tolist()
                for i in range(len(output_token_ids)):
                    output_token_ids[i] = output_token_ids[i][:][input_length[i] :]
                    # Find the first occurrence of
                    # an EOS token in the sequence
                    first_eos_idx = next(
                        (
                            idx
                            for idx, token_id in enumerate(output_token_ids[i])
                            if token_id in eos_token_id
                        ),
                        None,
                    )
                    # If an EOS token is found, only the previous
                    # part of it is retained (including the EOS token)
                    if first_eos_idx is not None:
                        output_token_ids[i] = output_token_ids[i][: first_eos_idx + 1]

                response = self.tokenizer.batch_decode(output_token_ids, skip_special_tokens=True)
                if not batched:
                    response = response[0]
                    output_token_ids = output_token_ids[0]
                yield ModelStatusCode.STREAM_ING, response, output_token_ids
                # stop when each sentence is finished,
                # or if we exceed the maximum length
                if unfinished_sequences.max() == 0 or stopping_criteria(input_ids, scores):
                    break
            yield ModelStatusCode.END, response, output_token_ids

    def stream_chat(
        self,
        inputs: List[dict],
        do_sample: bool = True,
        **kwargs,
    ):
        """Return the chat completions in stream mode.

        Args:
            inputs (List[dict]): input messages to be completed.
            do_sample (bool): do sampling if enabled
        Returns:
            the text/chat completion
        """
        prompt = self.template_parser(inputs)
        yield from self.stream_generate(prompt, do_sample, **kwargs)


class HFTransformerCasualLM(HFTransformer):

    def _load_model(self, path: str, model_kwargs: dict):
        import torch
        from transformers import AutoModelForCausalLM

        model_kwargs.setdefault('torch_dtype', torch.float16)
        self.model = AutoModelForCausalLM.from_pretrained(
            path, trust_remote_code=True, **model_kwargs
        )
        self.model.eval()


class HFTransformerChat(HFTransformerCasualLM):

    def __init__(self, template_parser=APITemplateParser, **kwargs):
        super().__init__(template_parser=template_parser, **kwargs)

    def chat(self, inputs: Union[List[dict], List[List[dict]]], do_sample: bool = True, **kwargs):
        """Return the chat completions in stream mode.

        Args:
            inputs (Union[List[dict], List[List[dict]]]): input messages to be completed.
            do_sample (bool): do sampling if enabled
        Returns:
            the text/chat completion
        """
        # handle batch inference with vanilla for loop
        if isinstance(inputs[0], list):
            resps = []
            for input in inputs:
                resps.append(self.chat(input, do_sample, **kwargs))
            return resps
        prompt = self.template_parser(inputs)
        query = prompt[-1]['content']
        history = prompt[:-1]
        try:
            response, history = self.model.chat(self.tokenizer, query, history=history)
        except Exception as e:
            # handle over-length input error
            logger.warning(str(e))
            response = ''
        return response
