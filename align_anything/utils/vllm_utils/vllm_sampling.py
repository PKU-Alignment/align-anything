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
"""vLLM sampling implementation for accelerated text generation."""

import logging
from typing import List, Tuple

import ray
import torch
import torch.distributed as dist
from vllm import SamplingParams


logger = logging.getLogger(__name__)


def generate_with_vllm(
    prompts: List[str],
    tokenizer,
    vllm_engines: List,
    n_samples_per_prompt: int = 1,
    temperature: float = 1.0,
    top_p: float = 1.0,
    top_k: int = -1,
    max_new_tokens: int = 1024,
    min_new_tokens: int = 1,
    skip_special_tokens: bool = False,
    **kwargs,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate text using vLLM.

    Args:
        prompts: List of prompts to generate from.
        tokenizer: Tokenizer to use.
        vllm_engines: List of vLLM engines.
        n_samples_per_prompt: Number of samples to generate per prompt.
        max_length: Maximum length of generated text.
        temperature: Temperature for sampling.
        top_p: Top-p for sampling.
        top_k: Top-k for sampling.
        max_new_tokens: Maximum number of new tokens to generate.
        min_new_tokens: Minimum number of new tokens to generate.
        skip_special_tokens: Whether to skip special tokens.
        **kwargs: Additional arguments.

    Returns:
        Tuple of (sequences, attention_mask, action_mask).
    """
    # round-robin load balance
    rank = dist.get_rank()

    llms = [vllm_engines[0]]

    print('creating sampling params...')

    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        max_tokens=max_new_tokens,
        min_tokens=min_new_tokens,
        skip_special_tokens=skip_special_tokens,
        include_stop_str_in_output=True,
    )

    # Expand prompt list based on the number of samples per prompt
    all_raw_prompts = sum([[prompt] * n_samples_per_prompt for prompt in prompts], [])

    all_prompts = []
    for raw_prompt in all_raw_prompts:
        all_prompts.append([{'role': 'user', 'content': raw_prompt}])

    print('distributing requests to engines...')

    # Distribute requests to engines and collect responses to outputs
    refs = []
    batch_size = (len(all_prompts) + len(llms) - 1) // len(llms)
    for i, llm in enumerate(llms):
        prompts = all_prompts[i * batch_size : (i + 1) * batch_size]
        refs.append(llm.add_requests.remote(rank, sampling_params=sampling_params, prompts=prompts))
    ray.get(refs)

    # Retrieve and combine results from all outputs
    all_output_refs = []
    for i, llm in enumerate(llms):
        all_output_refs.append(llm.get_responses.remote(rank))
    all_outputs = sum(ray.get(all_output_refs), [])
    # Process outputs in micro batches
    all_sequences = []
    all_attention_masks = []
    max_input_len, max_output_len = 0, 0
    pad_token_id, eos_token_id = tokenizer.pad_token_id, tokenizer.eos_token_id

    print('all_outputs', len(all_outputs))  # 32 1

    for i in range(0, len(all_outputs)):
        current_outputs = all_outputs[i]
        max_input_len = max(max_input_len, len(current_outputs.prompt_token_ids))
        max_output_len = max(max_output_len, len(current_outputs.outputs[0].token_ids))

    for i in range(0, len(all_outputs)):
        current_outputs = all_outputs[i]
        response = current_outputs.outputs[0]

        # left padding input
        input_len = len(current_outputs.prompt_token_ids)
        input_ids = [pad_token_id] * (max_input_len - input_len) + list(
            current_outputs.prompt_token_ids
        )

        # right padding output
        output_len = len(response.token_ids)
        output_ids = list(response.token_ids) + [pad_token_id] * (max_output_len - output_len)

        # concat input and output
        sequences = torch.tensor(input_ids + output_ids, device='cuda').unsqueeze(0)

        # Create attention mask (1 for tokens, 0 for padding)
        attention_mask = torch.logical_and(
            sequences.ne(pad_token_id), sequences.ne(eos_token_id)
        ).long()

        all_sequences.append(sequences)
        all_attention_masks.append(attention_mask)

    if all_sequences:
        sequences = torch.cat(all_sequences, dim=0).to(torch.long)
        attention_mask = torch.cat(all_attention_masks, dim=0).to(torch.bool)
        return sequences, attention_mask
    else:
        # Return empty tensors if no outputs
        return torch.tensor([], device='cuda', dtype=torch.long), torch.tensor(
            [], device='cuda', dtype=torch.bool
        )
