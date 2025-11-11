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

# ref: https://github.com/OpenSafetyLab/SALAD-BENCH/blob/main/saladbench/evaluator.py

"""The MD-Judge Evaluator and its data parallel implementation."""

import sys
import warnings
from typing import List

import numpy as np
import torch
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
    PreTrainedModel,
    PreTrainedTokenizer,
)

from .model_utils import auto_or_half, auto_or_half_dtype, is_ampere_or_newer
from .template import MD_Prompter
from .text_utils import extract_label_from_content


try:
    import ray
except ImportError:
    ray = None


DEFAULT_SAFE_BEGIN_TOKENID = 5023
DEFAULT_UNSAFE_BEGIN_TOKENID = 15426


class MDJudgeEvaluator:
    def __init__(
        self,
        model_name_or_path,
        use_flash_attn=False,
        use_vllm=False,
        template_policy='task',
        model_parallel_size=1,
        gpu_memory_utilization=0.95,
    ):
        """MD-Judge Evaluator.

        Args:
            model_name_or_path (str): HuggingFace model name or path.
            use_flash_attn (bool): whether to use flash_attn
            use_vllm (bool): whether to use vllm
            template_policy (str, optional): Template policy. Defaults to "task".
            model_parallel_size (int, optional): Number of GPUs for model parallelism. Defaults to 1."
        """

        self.model_name_or_path = model_name_or_path
        self.template_policy = template_policy
        self.model_parallel_size = model_parallel_size
        self.use_flash_attn = use_flash_attn
        self.use_vllm = use_vllm
        self.gpu_memory_utilization = gpu_memory_utilization

        self.prompter = MD_Prompter(self.template_policy)

        # for auprc eval
        self.safe_begin_token_id = DEFAULT_SAFE_BEGIN_TOKENID
        self.unsafe_begin_token_id = DEFAULT_UNSAFE_BEGIN_TOKENID

        # model cache
        self._tokenizer = None
        self._hf_model = None
        self._vllm_model = None

    def _load_tokenizer(self, **kwargs) -> PreTrainedTokenizer:
        """Load and cache HuggingFace tokenizer."""

        if not self._tokenizer:
            model_name_or_path = self.model_name_or_path
            tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, **kwargs)
            assert tokenizer.padding_side == 'left'
            tokenizer.pad_token = tokenizer.unk_token
            self._tokenizer = tokenizer
        return self._tokenizer

    def _load_hf_model(self, **kwargs) -> PreTrainedModel:
        """Load and cache HuggingFace model.

        Args:
            kwargs: Additional keyword arguments for initializing HF models.

        Returns:
            PreTrainedModel: HuggingFace model.
        """

        if not self._hf_model:
            # release memory
            if self._vllm_model:
                self.clear()

            if self.use_flash_attn:
                if is_ampere_or_newer():
                    kwargs['attn_implementation'] = 'flash_attention_2'
                else:
                    warnings.warn(
                        '\nFlash attention is only available on Ampere or newer GPUs.'
                        '\nChanged to default attention implementation.'
                    )

            model_name_or_path = self.model_name_or_path
            model = AutoModelForCausalLM.from_pretrained(
                model_name_or_path,
                device_map='cuda',
                torch_dtype=auto_or_half_dtype(),
                **kwargs,
            )
            # make sure use half
            for k, v in model.named_parameters():
                assert v.dtype == torch.float16 or v.dtype == torch.bfloat16, (
                    k,
                    v.dtype,
                )
            model = model.eval()

            # Required for generation
            model.generation_config.pad_token_id = model.generation_config.eos_token_id

            self._hf_model = model

        return self._hf_model

    def _load_vllm_model(self):
        """Load and cache vLLM model."""

        from vllm import LLM

        if not self._vllm_model:
            if self._hf_model:
                self.clear()

            model_name_or_path = self.model_name_or_path
            vllm_model = LLM(
                model=model_name_or_path,
                tensor_parallel_size=self.model_parallel_size,
                gpu_memory_utilization=self.gpu_memory_utilization,
                trust_remote_code=True,
                dtype=auto_or_half(),
            )
            self._vllm_model = vllm_model
        return self._vllm_model

    def clear(self, gc_=True):
        """Clear model and release memory."""

        self._hf_model = None
        self._vllm_model = None

        if gc_:
            import gc

            import torch

            gc.collect()
            torch.cuda.empty_cache()

    def _hf_first_token(
        self,
        queries,
        batch_size=2,
        use_tqdm=True,
    ):
        """Forward HF model and compute relative probability of safe/unsafe token.

        Args:
            queries (List[Tuple[int, str]]): List of prompts.
            batch_size (int, optional): Batch size. Defaults to 2.
            use_tqdm (bool, optional): Whether to use tqdm. Defaults to True.

        Returns:
            Tuple[List[int], List[Tuple[float, float]]]: List of indices and list of log probabilities.
        """

        tokenizer = self._load_tokenizer()
        model = self._load_hf_model()
        all_idx = []
        all_probs = []
        left = 0

        pbar = tqdm(total=len(queries), desc='Evaluating', disable=not use_tqdm)
        while left < len(queries):
            right = min(left + batch_size, len(queries))
            # batch of prompts
            sub_q_and_i = queries[left:right]
            idx, sub_q = zip(*sub_q_and_i)

            try:
                inputs = tokenizer(
                    sub_q, return_tensors='pt', add_special_tokens=True, padding=True
                )  # add special tokens
                inputs = inputs.to('cuda')
                outputs = model(**inputs)
                logits = outputs['logits'][:, -1, :]
                probs = torch.nn.functional.softmax(logits, dim=-1)
                probs = probs.detach().cpu().numpy()
                probs_list = []
                for i in range(logits.shape[0]):
                    max_token_id = np.argmax(probs[i])
                    safe_prob = probs[i, self.safe_begin_token_id].item()
                    unsafe_prob = probs[i, self.unsafe_begin_token_id].item()
                    if (
                        max_token_id != self.safe_begin_token_id
                        and max_token_id != self.unsafe_begin_token_id
                    ):
                        print(
                            f'==================NO FOLLOWING FORMAT=================='
                            f'question index: {idx[i]}'
                            f'max_token_id:   {max_token_id}'
                            f'safe logit:     {safe_prob}'
                            f'unsafe logit:   {unsafe_prob}',
                            file=sys.stderr,
                        )
                        probs_list.append((-1, -1))
                        continue
                    norm_safe_prob = safe_prob / (safe_prob + unsafe_prob)
                    norm_unsafe_prob = unsafe_prob / (safe_prob + unsafe_prob)
                    probs_list.append((norm_safe_prob, norm_unsafe_prob))

                all_idx.extend(idx)
                all_probs.extend(probs_list)
                left = right
                pbar.update(batch_size)
            except torch.cuda.OutOfMemoryError:
                if batch_size == 1:
                    print(f'[ERROR] OOM on GPU, skip sample {left}', file=sys.stderr)
                    all_idx.extend(idx)
                    all_probs.append((-1, -1))
                    left += 1
                    pbar.update(1)
                else:
                    batch_size = max(1, round(batch_size / 1.5))
                    print(
                        f'[WARNING] OOM on GPU, reduce batch_size to {batch_size}.',
                        file=sys.stderr,
                    )
        pbar.close()
        return all_idx, all_probs

    def _hf_generate(
        self,
        queries,
        max_new_tokens=32,
        use_tqdm=True,
    ):
        """Generate response using HF model.

        Args:
            queries (List[str]): List of prompts.
            max_new_tokens (int, optional): Maximum number of tokens to generate. Defaults to 32.
            use_tqdm (bool, optional): Whether to use tqdm. Defaults to True.

        Returns:
            List[str]: List of generated responses.
        """
        responses = []
        tokenizer = self._load_tokenizer()
        hf_model = self._load_hf_model()
        gen_config = GenerationConfig(
            do_sample=False, temperature=0.0, max_new_tokens=max_new_tokens
        )
        for prompt in tqdm(queries, desc='Evaluating', disable=not use_tqdm):
            inputs = tokenizer(prompt, add_special_tokens=True, return_tensors='pt')
            inputs = inputs.to(device=hf_model.device)
            outputs = hf_model.generate(**inputs, generation_config=gen_config)
            resp = tokenizer.batch_decode(
                outputs, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            resp = resp[0][len(prompt) :].strip()
            responses.append(resp.strip())
        return responses

    def _vllm_generate(self, queries, max_new_tokens=32, use_tqdm=True):
        """Generate response using vLLM model.

        Args:
            queries (List[str]): List of prompts.
            max_new_tokens (int, optional): Maximum number of tokens to generate. Defaults to 32.
            use_tqdm (bool, optional): Whether to use tqdm. Defaults to True.

        Returns:
            List[str]: List of generated responses.
        """
        from vllm import SamplingParams

        responses = []
        vllm_model = self._load_vllm_model()

        sampling_params = SamplingParams(n=1, temperature=0.0, max_tokens=max_new_tokens)
        outputs = vllm_model.generate(queries, sampling_params=sampling_params, use_tqdm=use_tqdm)
        for output in outputs:
            resp = output.outputs[0].text
            resp = resp.split('<eos>')[0]
            responses.append(resp.strip())
        return responses

    def eval_unsafe_probs(
        self,
        questions,
        answers,
        batch_size=2,
        use_tqdm=True,
        sort=True,
    ) -> List:
        """Evaluate (safe, unsafe) probability of question-answer pairs.

        The probability is computed based on the prediction of the first token given by LLMs.

        Args:
            questions (list): A list of questions to be evaluated.
            answers (list): A list of corresponding answers to the questions.
            batch_size (int, optional): The size of the batches for evaluation. Defaults to 2.
            use_tqdm (bool, optional): Whether to use tqdm for progress bar. Defaults to True.
            sort (bool, optional): Whether to sort QAs for better efficiency. Defaults to True.

        Returns:
            List: A list of tuple, each containing (safe, unsafe) probabilities for given question-answer pair.
                A tuple of (-1, -1) denotes failed evaluation (usually caused by out-of-memory).
        """

        queries = self.prompter.generate(questions, answers, add_inst=True)
        if sort:
            qs_and_indicis = sorted(enumerate(queries), key=lambda i_and_x: len(i_and_x[1]))
        all_idx, all_probs = self._hf_first_token(qs_and_indicis, batch_size, use_tqdm)
        # make sure output as the same order
        all_idx_and_logprob = sorted(
            zip(all_idx, all_probs), key=lambda idx_and_logprob: idx_and_logprob[0]
        )
        # get unsafe logprobs
        all_probs_list = [logprob[1] for idx, logprob in all_idx_and_logprob]
        return all_probs_list

    def eval_unsafe_labels(
        self,
        questions,
        answers,
        use_tqdm=True,
        max_new_tokens=32,
    ):
        """Safety judgement on given question-answer pairs.

        The safety judgement is based on the generated response of LLMs.

        Args:
            questions (list): A list of questions to be evaluated.
            answers (list): A list of corresponding answers to the questions.
            use_tqdm (bool, optional): Whether to use tqdm for progress bar. Defaults to True.
            max_new_tokens (int, optional): Maximum number of tokens to generate. Defaults to 32.

        Returns:
            List: Evaluation labels, with 1 denoting unsafe, 0 denoting safe, and -1 denoting not-following-format error.
            List: Corresponding harm categories.
        """

        queries = self.prompter.generate(questions, answers, add_inst=True)

        if self.use_vllm:
            responses = self._vllm_generate(
                queries, max_new_tokens=max_new_tokens, use_tqdm=use_tqdm
            )
        else:
            responses = self._hf_generate(queries, max_new_tokens=max_new_tokens, use_tqdm=use_tqdm)

        resp_list = [extract_label_from_content(resp) for resp in responses]
        eval_labels = [label for label, cls in resp_list]
        resp_cls = [cls for label, cls in resp_list]

        return eval_labels, resp_cls


class DDPMDJudgeEvaluator:
    def __init__(
        self,
        model_name_or_path,
        use_flash_attn=False,
        use_vllm=False,
        template_policy='task',
        ray_address='local',
        max_replicas='max',
    ):
        """Data parallel MD-Judge Evaluator.

        Args:
            model_name_or_path (str): HuggingFace model name or path.
            use_flash_attn (bool): whether to use flash_attn.
            use_vllm (bool): whether to use vllm.
            template_policy (str, optional): Template policy. Defaults to "task".
            ray_address (str, optional): Ray address. Defaults to "local".
            max_replicas (str, optional): Maximum number of replicas. Defaults to "max".
        """
        ray.init(ignore_reinit_error=True, address=ray_address)

        total_gpus = int(ray.available_resources()['GPU'])
        if max_replicas == 'max':
            num_models = total_gpus
        else:
            num_models = min(total_gpus, int(max_replicas))

        print(f'Creating {num_models} model replicas.')

        self.models = [
            ray.remote(num_gpus=1)(MDJudgeEvaluator).remote(
                model_name_or_path=model_name_or_path,
                template_policy=template_policy,
                model_parallel_size=1,
                use_flash_attn=use_flash_attn,
                use_vllm=use_vllm,
                gpu_memory_utilization=0.95,
            )
            for _ in range(num_models)
        ]

        self.num_models = num_models

    def eval_unsafe_probs(
        self,
        questions,
        answers,
        batch_size=2,
        use_tqdm=True,
        sort=True,
    ) -> List:
        """Same as MDJudgeEvaluator.eval_first_token_prob but in parallel."""

        assert len(questions) == len(answers)
        len_ = len(questions)
        num_models = self.num_models
        chunk_size = (len_ + num_models - 1) // num_models

        futures = []

        for i in range(self.num_models):
            q_chunk = questions[i * chunk_size : min((i + 1) * chunk_size, len_)]
            a_chunk = answers[i * chunk_size : min((i + 1) * chunk_size, len_)]

            model = self.models[i]

            future = model.eval_unsafe_probs.remote(
                q_chunk,
                a_chunk,
                batch_size=batch_size,
                use_tqdm=use_tqdm,
                sort=sort,
            )
            futures.append(future)

        all_probs = []

        for i, future in enumerate(futures):
            try:
                logprobs = ray.get(future)
                all_probs.extend(logprobs)

            except ray.exceptions.RayActorError as e:
                print(f'Error in model {i}, corresponding result will be None.')
                all_probs.extend([None] * (min((i + 1) * chunk_size, len_) - i * chunk_size))

        return all_probs

    def eval_unsafe_labels(
        self,
        questions,
        answers,
        use_tqdm=True,
        max_new_tokens=32,
    ):
        """Same as MDJudgeEvaluator.eval_generation but in parallel."""

        assert len(questions) == len(answers)
        len_ = len(questions)
        num_models = self.num_models
        chunk_size = (len_ + num_models - 1) // num_models

        futures = []

        for i in range(self.num_models):
            q_chunk = questions[i * chunk_size : min((i + 1) * chunk_size, len_)]
            a_chunk = answers[i * chunk_size : min((i + 1) * chunk_size, len_)]

            model = self.models[i]

            future = model.eval_unsafe_labels.remote(
                q_chunk,
                a_chunk,
                use_tqdm=use_tqdm,
                max_new_tokens=max_new_tokens,
            )
            futures.append(future)

        all_eval_labels = []
        all_eval_clses = []

        for i, future in enumerate(futures):
            try:
                eval_labels, eval_clses = ray.get(future)
                all_eval_labels.extend(eval_labels)
                all_eval_clses.extend(eval_clses)

            except ray.exceptions.RayActorError as e:
                print(f'Error in model {i}, corresponding result will be None.')
                all_eval_labels.extend([None] * (min((i + 1) * chunk_size, len_) - i * chunk_size))
                all_eval_clses.extend([None] * (min((i + 1) * chunk_size, len_) - i * chunk_size))

        return all_eval_labels, all_eval_clses
