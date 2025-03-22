# Copyright 2024 Allen Institute for AI

# Copyright 2025 Align-Anything Team. All Rights Reserved.
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


from collections import OrderedDict
from typing import Dict, Optional, Tuple, cast

import torch
from allenact.algorithms.onpolicy_sync.losses import PPO
from allenact.algorithms.onpolicy_sync.losses.abstract_loss import (
    AbstractActorCriticLoss,
    ObservationType,
)
from allenact.base_abstractions.distributions import CategoricalDistr, Distr
from allenact.base_abstractions.misc import ActorCriticOutput


class Imitation(AbstractActorCriticLoss):
    """Expert imitation loss."""

    def __init__(self, uuid: str = 'expert_pickupable', action_idx: int = 8, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.uuid = uuid
        self.action_idx = action_idx

    def loss(  # type: ignore
        self,
        step_count: int,
        batch: ObservationType,
        actor_critic_output: ActorCriticOutput[Distr],
        *args,
        **kwargs,
    ):
        """Computes the imitation loss.

        # Parameters

        batch : A batch of data corresponding to the information collected when rolling out (possibly many) agents
            over a fixed number of steps. In particular this batch should have the same format as that returned by
            `RolloutStorage.batched_experience_generator`.
            Here `batch["observations"]` must contain `"expert_action"` observations
            or `"expert_policy"` observations. See `ExpertActionSensor` (or `ExpertPolicySensor`) for an example of
            a sensor producing such observations.
        actor_critic_output : The output of calling an ActorCriticModel on the observations in `batch`.
        args : Extra args. Ignored.
        kwargs : Extra kwargs. Ignored.

        # Returns

        A (0-dimensional) torch.FloatTensor corresponding to the computed loss. `.backward()` will be called on this
        tensor in order to compute a gradient update to the ActorCriticModel's parameters.
        """
        observations = cast(Dict[str, torch.Tensor], batch['observations'])

        losses = OrderedDict()

        should_report_loss = False
        has_observation_to_compute = False

        total_loss = 0
        if self.uuid in observations:
            should_report_loss = True
            has_observation_to_compute = True
            total_loss += torch.nn.functional.binary_cross_entropy_with_logits(
                actor_critic_output.distributions.logits[:, :, self.action_idx],
                observations[self.uuid],
            )

        if not has_observation_to_compute:
            raise NotImplementedError(
                'Imitation loss requires either `expert_action` or `expert_policy`'
                ' sensor to be active.'
            )
        return (
            total_loss,
            {'expert_cross_entropy': total_loss.item(), **losses} if should_report_loss else {},
        )


class PPOValueStopGrad(AbstractActorCriticLoss):
    """Implementation of the Proximal Policy Optimization loss.

    # Attributes

    clip_param : The clipping parameter to use.
    use_clipped_value_loss : Whether or not to also clip the value loss.
    """

    def __init__(
        self,
        clip_param: float,
        discrete_critics: bool,
        use_clipped_value_loss=True,
        clip_decay=None,
        *args,
        **kwargs,
    ):
        """Initializer.

        See the class documentation for parameter definitions.
        """
        super().__init__(*args, **kwargs)
        self.clip_param = clip_param
        self.use_clipped_value_loss = use_clipped_value_loss
        self.clip_decay = clip_decay if clip_decay is not None else (lambda x: 1.0)
        self.discrete_critics = discrete_critics

    def loss(  # type: ignore
        self,
        step_count: int,
        batch: ObservationType,
        actor_critic_output: ActorCriticOutput[CategoricalDistr],
        *args,
        **kwargs,
    ):
        if self.discrete_critics:
            logits = actor_critic_output.extras['stop_grad_logits']
            loss_func = actor_critic_output.extras['loss_func']
            reshaped_logits = logits.view(-1, logits.shape[-1])
            returns = cast(torch.FloatTensor, batch['returns'])
            reshaped_returns = returns.view(-1)
            value_loss = 0.5 * loss_func(reshaped_logits, reshaped_returns)
        else:
            values = actor_critic_output.extras['stop_grad_values']
            clip_param = self.clip_param * self.clip_decay(step_count)
            if self.use_clipped_value_loss:
                value_pred_clipped = batch['values'] + (values - batch['values']).clamp(
                    -clip_param, clip_param
                )
                value_losses = (values - batch['returns']).pow(2)
                value_losses_clipped = (value_pred_clipped - batch['returns']).pow(2)
                value_loss = 0.5 * torch.max(value_losses, value_losses_clipped).mean()
            else:
                value_loss = (
                    0.5 * (cast(torch.FloatTensor, batch['returns']) - values).pow(2).mean()
                )

        bias_norm = actor_critic_output.extras['bias_norm']
        weight_norm = actor_critic_output.extras['weight_norm']
        try:
            weight_grad = actor_critic_output.extras['weight_grad_norm']
        except:
            weight_grad = 0

        return (
            value_loss,
            {
                'value': value_loss.item(),
                'bias_norm': bias_norm,
                'weight_norm': weight_norm,
                'weight_grad': weight_grad,
            },
        )


class PPOLogGrad(PPO):
    def __init__(self, discrete_critics: bool, action_loss_schedule, *args, **kwargs):
        """
        Args:
            discrete_critics: whether the critic is discrete
            action_loss_schedule: a function that takes the step count and returns the weight for the action loss
        """
        super().__init__(*args, **kwargs)
        self.discrete_critics = discrete_critics
        self.action_loss_schedule = (
            action_loss_schedule if action_loss_schedule is not None else (lambda x: 1.0)
        )

    def loss_per_step(
        self,
        step_count: int,
        batch: ObservationType,
        actor_critic_output: ActorCriticOutput[CategoricalDistr],
    ) -> Tuple[
        Dict[str, Tuple[torch.Tensor, Optional[float]]], Dict[str, torch.Tensor]
    ]:  # TODO tuple output

        actions = cast(torch.LongTensor, batch['actions'])

        action_log_probs = actor_critic_output.distributions.log_prob(actions)
        dist_entropy: torch.FloatTensor = getattr(
            actor_critic_output.distributions, self.entropy_method_name
        )()

        def add_trailing_dims(t: torch.Tensor):
            assert len(t.shape) <= len(batch[self.adv_key].shape)
            return t.view(t.shape + ((1,) * (len(batch[self.adv_key].shape) - len(t.shape))))

        dist_entropy = add_trailing_dims(dist_entropy)

        clip_param = self.clip_param * self.clip_decay(step_count)

        ratio = torch.exp(action_log_probs - batch['old_action_log_probs'])
        ratio = add_trailing_dims(ratio)
        clamped_ratio = torch.clamp(ratio, 1.0 - clip_param, 1.0 + clip_param)

        surr1 = ratio * batch[self.adv_key]
        surr2 = clamped_ratio * batch[self.adv_key]

        use_clamped = surr2 < surr1
        action_loss = -torch.where(cast(torch.Tensor, use_clamped), surr2, surr1)

        if self.discrete_critics:
            logits = actor_critic_output.extras['full_logits']
            loss_func = actor_critic_output.extras['loss_func']
            reshaped_logits = logits.view(-1, logits.shape[-1])
            returns = cast(torch.FloatTensor, batch['returns'])
            reshaped_returns = returns.view(-1)
            value_loss = 0.5 * loss_func(reshaped_logits, reshaped_returns)
        else:
            values = actor_critic_output.values
            clip_param = self.clip_param * self.clip_decay(step_count)
            if self.use_clipped_value_loss:
                value_pred_clipped = batch['values'] + (values - batch['values']).clamp(
                    -clip_param, clip_param
                )
                value_losses = (values - batch['returns']).pow(2)
                value_losses_clipped = (value_pred_clipped - batch['returns']).pow(2)
                value_loss = 0.5 * torch.max(value_losses, value_losses_clipped).mean()
            else:
                value_loss = (
                    0.5 * (cast(torch.FloatTensor, batch['returns']) - values).pow(2).mean()
                )

        bias_norm = actor_critic_output.extras['bias_norm']
        weight_norm = actor_critic_output.extras['weight_norm']
        try:
            weight_grad = actor_critic_output.extras['weight_grad_norm']
        except:
            weight_grad = torch.tensor([0.0])

        action_weight = self.action_loss_schedule(step_count)

        # noinspection PyUnresolvedReferences
        return (
            {
                'value': (value_loss, self.value_loss_coef),
                'action': (action_loss, action_weight),
                'entropy': (dist_entropy.mul_(-1.0), self.entropy_coef),  # type: ignore
            },
            {
                'bias_norm': bias_norm,
                'weight_norm': weight_norm,
                'weight_grad': weight_grad,
                'action_weight': action_weight,
                # "ratio": ratio,
                # "ratio_clamped": clamped_ratio,
                # "ratio_used": torch.where(
                #     cast(torch.Tensor, use_clamped), clamped_ratio, ratio
                # ),
            },
        )

    def loss(  # type: ignore
        self,
        step_count: int,
        batch: ObservationType,
        actor_critic_output: ActorCriticOutput[CategoricalDistr],
        *args,
        **kwargs,
    ):
        losses_per_step, ratio_info = self.loss_per_step(
            step_count=step_count,
            batch=batch,
            actor_critic_output=actor_critic_output,
        )
        losses = {key: (loss.mean(), weight) for (key, (loss, weight)) in losses_per_step.items()}

        total_loss = sum(
            loss * weight if weight is not None else loss for loss, weight in losses.values()
        )

        result = (
            total_loss,
            {
                'ppo_total': cast(torch.Tensor, total_loss).item(),
                **{key: loss.item() for key, (loss, _) in losses.items()},
            }
            | ratio_info,
        )

        return result


class SafePPOLogGrad(PPO):
    def __init__(self, discrete_critics: bool, action_loss_schedule, *args, **kwargs):
        """
        Args:
            discrete_critics: whether the critic is discrete
            action_loss_schedule: a function that takes the step count and returns the weight for the action loss
        """
        super().__init__(*args, **kwargs)
        self.discrete_critics = discrete_critics
        self.action_loss_schedule = (
            action_loss_schedule if action_loss_schedule is not None else (lambda x: 1.0)
        )

    def loss_per_step(
        self,
        step_count: int,
        batch: ObservationType,
        actor_critic_output: ActorCriticOutput[CategoricalDistr],
        lagrangian_multiplier: torch.Tensor,
    ) -> Tuple[
        Dict[str, Tuple[torch.Tensor, Optional[float]]], Dict[str, torch.Tensor]
    ]:  # TODO tuple output

        actions = cast(torch.LongTensor, batch['actions'])

        action_log_probs = actor_critic_output.distributions.log_prob(actions)
        dist_entropy: torch.FloatTensor = getattr(
            actor_critic_output.distributions, self.entropy_method_name
        )()

        def add_trailing_dims(t: torch.Tensor):
            assert len(t.shape) <= len(batch[self.adv_key].shape)
            return t.view(t.shape + ((1,) * (len(batch[self.adv_key].shape) - len(t.shape))))

        dist_entropy = add_trailing_dims(dist_entropy)

        clip_param = self.clip_param * self.clip_decay(step_count)

        ratio = torch.exp(action_log_probs - batch['old_action_log_probs'])
        ratio = add_trailing_dims(ratio)
        clamped_ratio = torch.clamp(ratio, 1.0 - clip_param, 1.0 + clip_param)

        penalty = lagrangian_multiplier.item()
        surr1 = ratio * (batch[self.adv_key] - penalty * batch['adv_c_key']) / (1 + penalty)
        surr2 = clamped_ratio * (batch[self.adv_key] - penalty * batch['adv_c_key']) / (1 + penalty)

        use_clamped = surr2 < surr1
        action_loss = -torch.where(cast(torch.Tensor, use_clamped), surr2, surr1)

        if self.discrete_critics:
            logits = actor_critic_output.extras['full_logits']
            loss_func = actor_critic_output.extras['loss_func']
            reshaped_logits = logits.view(-1, logits.shape[-1])
            returns = cast(torch.FloatTensor, batch['returns'])
            reshaped_returns = returns.view(-1)
            value_loss = 0.5 * loss_func(reshaped_logits, reshaped_returns)
        else:
            values = actor_critic_output.values
            clip_param = self.clip_param * self.clip_decay(step_count)
            if self.use_clipped_value_loss:
                value_pred_clipped = batch['values'] + (values - batch['values']).clamp(
                    -clip_param, clip_param
                )
                value_losses = (values - batch['returns']).pow(2)
                value_losses_clipped = (value_pred_clipped - batch['returns']).pow(2)
                value_loss = 0.5 * torch.max(value_losses, value_losses_clipped).mean()
            else:
                value_loss = (
                    0.5 * (cast(torch.FloatTensor, batch['returns']) - values).pow(2).mean()
                )

        bias_norm = actor_critic_output.extras['bias_norm']
        weight_norm = actor_critic_output.extras['weight_norm']
        try:
            weight_grad = actor_critic_output.extras['weight_grad_norm']
        except:
            weight_grad = torch.tensor([0.0])

        action_weight = self.action_loss_schedule(step_count)

        # noinspection PyUnresolvedReferences
        return (
            {
                'value': (value_loss, self.value_loss_coef),
                'action': (action_loss, action_weight),
                'entropy': (dist_entropy.mul_(-1.0), self.entropy_coef),  # type: ignore
            },
            {
                'bias_norm': bias_norm,
                'weight_norm': weight_norm,
                'weight_grad': weight_grad,
                'action_weight': action_weight,
                # "ratio": ratio,
                # "ratio_clamped": clamped_ratio,
                # "ratio_used": torch.where(
                #     cast(torch.Tensor, use_clamped), clamped_ratio, ratio
                # ),
            },
        )

    def loss(  # type: ignore
        self,
        step_count: int,
        batch: ObservationType,
        actor_critic_output: ActorCriticOutput[CategoricalDistr],
        *args,
        **kwargs,
    ):
        losses_per_step, ratio_info = self.loss_per_step(
            step_count=step_count,
            batch=batch,
            actor_critic_output=actor_critic_output,
            lagrangian_multiplier=kwargs['lagrangian_multiplier'],
        )
        losses = {key: (loss.mean(), weight) for (key, (loss, weight)) in losses_per_step.items()}

        total_loss = sum(
            loss * weight if weight is not None else loss for loss, weight in losses.values()
        )

        result = (
            total_loss,
            {
                'ppo_total': cast(torch.Tensor, total_loss).item(),
                **{key: loss.item() for key, (loss, _) in losses.items()},
            }
            | ratio_info,
        )

        return result


class PPOStopGrad(PPO):
    def loss_per_step(
        self,
        step_count: int,
        batch: ObservationType,
        actor_critic_output: ActorCriticOutput[CategoricalDistr],
    ) -> Tuple[Dict[str, Tuple[torch.Tensor, Optional[float]]], Dict[str, torch.Tensor]]:

        actions = cast(torch.LongTensor, batch['actions'])
        # values = actor_critic_output.values
        values = actor_critic_output.extras['stop_grad_values']

        action_log_probs = actor_critic_output.distributions.log_prob(actions)
        dist_entropy: torch.FloatTensor = getattr(
            actor_critic_output.distributions, self.entropy_method_name
        )()

        def add_trailing_dims(t: torch.Tensor):
            assert len(t.shape) <= len(batch[self.adv_key].shape)
            return t.view(t.shape + ((1,) * (len(batch[self.adv_key].shape) - len(t.shape))))

        dist_entropy = add_trailing_dims(dist_entropy)

        clip_param = self.clip_param * self.clip_decay(step_count)

        ratio = torch.exp(action_log_probs - batch['old_action_log_probs'])
        ratio = add_trailing_dims(ratio)
        clamped_ratio = torch.clamp(ratio, 1.0 - clip_param, 1.0 + clip_param)

        surr1 = ratio * batch[self.adv_key]
        surr2 = clamped_ratio * batch[self.adv_key]

        use_clamped = surr2 < surr1
        action_loss = -torch.where(cast(torch.Tensor, use_clamped), surr2, surr1)

        if self.use_clipped_value_loss:
            value_pred_clipped = batch['values'] + (values - batch['values']).clamp(
                -clip_param, clip_param
            )
            value_losses = (values - batch['returns']).pow(2)
            value_losses_clipped = (value_pred_clipped - batch['returns']).pow(2)
            value_loss = 0.5 * torch.max(value_losses, value_losses_clipped)
        else:
            value_loss = 0.5 * (cast(torch.FloatTensor, batch['returns']) - values).pow(2)

        # noinspection PyUnresolvedReferences
        return (
            {
                'value': (value_loss, self.value_loss_coef),
                'action': (action_loss, None),
                'entropy': (dist_entropy.mul_(-1.0), self.entropy_coef),  # type: ignore
            },
            (
                {
                    'ratio': ratio,
                    'ratio_clamped': clamped_ratio,
                    'ratio_used': torch.where(
                        cast(torch.Tensor, use_clamped), clamped_ratio, ratio
                    ),
                }
                if self.show_ratios
                else {}
            ),
        )
