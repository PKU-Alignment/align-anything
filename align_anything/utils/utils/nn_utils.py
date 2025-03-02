# Copyright 2024 Allen Institute for AI

# Copyright 2024-2025 Align-Anything Team. All Rights Reserved.
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


from typing import Literal, List

import os
import torch
import torch.nn as nn

from align_anything.utils.utils.type_utils import THORActions
from allenact.utils.system import get_logger


def debug_model_info(model: nn.Module, trainable: bool = True, use_logger: bool = True, **kwargs):
    if int(os.environ.get("LOCAL_RANK", 0)) != 0:
        return
    debug_msg = (
        f"{model}"
        + (
            f"\nTrainable Parameters: "
            f"{sum(p.numel() for p in model.parameters() if p.requires_grad)}"
        )
        * trainable
    )
    if use_logger:
        get_logger().debug("".join([str(t) for t in debug_msg])[:-1], **kwargs)
    else:
        print(debug_msg, **kwargs)


def create_causal_mask(T: int, device: torch.device):
    return torch.triu(torch.full([T, T], float("-inf"), device=device), diagonal=1)


def sample_action_index_from_logits(
    logits: torch.Tensor,
    sampling: Literal[
        "greedy", "sample", "sample_done_only_if_argmax", "sample_done_only_if_prob_gt_thresh"
    ],
    action_list: List[str] = None,
) -> torch.Tensor:
    assert len(logits.shape) == 1, f"expected logits to be 1D, got {logits.shape}"
    if sampling == "greedy":
        action_idx = torch.argmax(logits, dim=-1)
    elif sampling == "sample":
        action_idx = torch.distributions.categorical.Categorical(logits=logits).sample()
    elif sampling == "sample_done_only_if_argmax":
        assert action_list is not None, f"action_list must be provided for {sampling}"
        action_idx = torch.distributions.categorical.Categorical(logits=logits).sample()
        # THORActions.done action is really "end"; but checking "done" too if we ever decide to make it "done"
        sampled_done = action_list[action_idx] in [THORActions.done, THORActions.sub_done]
        is_argmax = action_idx == torch.argmax(logits)
        if sampled_done and not is_argmax:
            while action_list[action_idx] in [THORActions.done, THORActions.sub_done]:
                action_idx = torch.distributions.categorical.Categorical(logits=logits).sample()
    elif sampling == "sample_done_only_if_prob_gt_thresh":
        assert action_list is not None, f"action_list must be provided for {sampling}"
        action_idx = torch.distributions.categorical.Categorical(logits=logits).sample()
        sampled_done = action_list[action_idx] in [THORActions.done, THORActions.sub_done]
        probs = torch.softmax(logits, dim=-1)
        is_gt_thresh = probs[action_idx] > 0.3
        if sampled_done and not is_gt_thresh:
            while action_list[action_idx] in [THORActions.done, THORActions.sub_done]:
                action_idx = torch.distributions.categorical.Categorical(logits=logits).sample()
    else:
        raise NotImplementedError(f"unknown sampling method {sampling}")

    return action_idx
