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
"""Baseline models for use in the object navigation task.

Object navigation is currently available as a Task in AI2-THOR and
Facebook's Habitat.
"""

import os
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple, Union, cast

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from allenact.algorithms.onpolicy_sync.policy import (
    DistributionType,
    LinearActorHead,
    LinearCriticHead,
    ObservationType,
)
from allenact.base_abstractions.misc import ActorCriticOutput, Memory
from allenact.embodiedai.aux_losses.losses import MultiAuxTaskNegEntropyLoss
from allenact.embodiedai.models.visual_nav_models import FusionType, VisualNavActorCritic
from allenact.utils.system import get_logger
from gym.spaces import Dict as SpaceDict
from transformers import AutoTokenizer, T5EncoderModel

from align_anything.models.spoc_models.models.llama.model import ModelArgs as LLAMAModelArgs
from align_anything.models.spoc_models.models.llama.model import (
    TransformerDecoder as LLAMATransformerDecoder,
)
from align_anything.models.spoc_models.models.transformer_models.text_cond_visual_encoder import (
    PositionalEncoder,
)
from align_anything.utils.spoc_utils.bbox_utils import get_best_of_two_bboxes
from align_anything.utils.spoc_utils.loss_functions import HLGaussLoss
from align_anything.utils.spoc_utils.nn_utils import debug_model_info
from align_anything.utils.spoc_utils.string_utils import convert_byte_to_string


TGTCache = torch.Tensor


class DinoLLAMATxNavActorCritic(VisualNavActorCritic):
    def __init__(
        # base params
        self,
        action_space: gym.spaces.Discrete,
        observation_space: SpaceDict,
        goal_sensor_uuid: str,
        hidden_size=512,
        num_tx_layers=3,
        num_tx_heads=8,
        text_embed_size=512,
        add_prev_actions=False,
        add_prev_action_null_token=False,
        action_embed_size=512,
        multiple_beliefs=False,
        linear_belief_adaptor=False,
        beliefs_fusion: Optional[FusionType] = None,
        auxiliary_uuids: Optional[List[str]] = None,
        # custom params
        rgb_dino_preprocessor_uuid: Optional[str] = None,
        manipulation_rgb_dino_preprocessor_uuid: Optional[str] = None,
        an_object_is_in_hand_uuid: Optional[str] = None,
        goal_dims: int = 512,
        dino_compressor_hidden_out_dims: Tuple[int, int] = (512, 512),
        combiner_hidden_out_dims: int = 512,
        combiner_nhead: int = 8,
        combiner_layers: int = 3,
        max_steps: int = 1000,
        max_steps_for_training: int = 128,
        time_step_uuid: Optional[str] = None,
        initial_tgt_cache_shape: Tuple[int, int, int] = (128, 32, 512),
        traj_idx_uuid: Optional[str] = None,
        traj_max_idx: Optional[int] = None,
        relevant_object_box_uuid: Optional[str] = None,
        accurate_object_box_uuid: Optional[str] = None,
        prev_checkpoint: Optional[str] = None,
        prev_rl_checkpoint: Optional[str] = None,
        critic_type='linear',
        **kwargs,
    ):
        super().__init__(
            action_space=action_space,
            observation_space=observation_space,
            hidden_size=hidden_size,
            multiple_beliefs=multiple_beliefs,
            beliefs_fusion=beliefs_fusion,
            auxiliary_uuids=auxiliary_uuids,
            **kwargs,
        )

        assert action_embed_size == combiner_hidden_out_dims
        self.time_step_counter = 0
        self.traj_idx_uuid = traj_idx_uuid
        self.traj_max_idx = traj_max_idx
        self.relevant_object_box_uuid = relevant_object_box_uuid
        self.accurate_object_box_uuid = accurate_object_box_uuid

        # self.done_action_idx = []

        self.text_embed_size = text_embed_size
        self.max_steps = max_steps
        self.max_steps_for_training = max_steps_for_training
        self.time_step_uuid = time_step_uuid
        self.goal_sensor_uuid = goal_sensor_uuid

        self.use_linear_belief_adaptor = linear_belief_adaptor
        if rgb_dino_preprocessor_uuid is not None:
            dino_preprocessor_uuid = rgb_dino_preprocessor_uuid
            self.visual_encoder = DinoTxGoalEncoder(
                self.observation_space,
                goal_sensor_uuid,
                dino_preprocessor_uuid,
                manipulation_rgb_dino_preprocessor_uuid=manipulation_rgb_dino_preprocessor_uuid,
                relevant_object_box_uuid=relevant_object_box_uuid,
                accurate_object_box_uuid=accurate_object_box_uuid,
                goal_embed_dims=goal_dims,
                dino_compressor_hidden_out_dims=dino_compressor_hidden_out_dims,
                combiner_hidden_out_dims=combiner_hidden_out_dims,
                combiner_heads=combiner_nhead,
                combiner_layers=combiner_layers,
            )

        self.an_object_is_in_hand_uuid = an_object_is_in_hand_uuid
        if an_object_is_in_hand_uuid:
            self.object_in_hand_embed = nn.Embedding(3, self.visual_encoder.output_dims)
            self.object_in_hand_embed.weight.data.uniform_(-0.01, 0.01)

        self.create_tx_state_encoders(
            obs_embed_size=self.visual_encoder.output_dims,
            text_embed_size=text_embed_size,
            num_tx_layers=num_tx_layers,
            num_tx_heads=num_tx_heads,
            add_prev_actions=add_prev_actions,
            add_prev_action_null_token=add_prev_action_null_token,
            prev_action_embed_size=action_embed_size,
            initial_tgt_cache_shape=initial_tgt_cache_shape,
        )

        # self.create_actorcritic_head()
        self.actor = LinearActorHead(self._hidden_size, self.action_space.n)
        self.critic_type = critic_type
        if critic_type == 'linear':
            self.critic = LinearCriticHead(self._hidden_size)
        elif critic_type == 'mlp':
            self.critic = MLPCriticHead(self._hidden_size)
        elif critic_type == 'discrete':
            # seems that bins = 101 is a good choice -> -5 to 15 = 20 / 100 = 0.2 width -> sigma=0.15
            dc_loss = HLGaussLoss(min_value=-5.0, max_value=15.0, num_bins=101, sigma=0.15)
            self.critic = DiscreteCriticHead(self._hidden_size, bin_size=101, loss_fn=dc_loss)
        else:
            print(f'Unknown critic type: {critic_type}')
            raise NotImplementedError

        self.create_aux_models(
            obs_embed_size=self.visual_encoder.output_dims,
            action_embed_size=action_embed_size,
        )

        if prev_checkpoint is not None:
            assert (
                prev_rl_checkpoint is None
            ), 'Cannot have both prev_checkpoint and prev_rl_checkpoint'
            # This is in charge of loading the pytorch lightning checkpoint form Imitation learning
            ckpt = prev_checkpoint  # self.get_ckpt_path(prev_checkpoint, ckpt_step)
            from utils.offline_train_utils import load_pl_ckpt_allenact

            load_pl_ckpt_allenact(self, ckpt)
        elif prev_rl_checkpoint is not None:
            ckpt = torch.load(os.path.abspath(prev_rl_checkpoint), map_location='cpu')

            ckpt = cast(
                Dict[str, Union[Dict[str, Any], torch.Tensor, float, int, str, List]],
                ckpt,
            )

            state_dict = ckpt['model_state_dict']
            state_dict = {k: v for k, v in state_dict.items() if 'critic_tsfm' not in k}
            load_status = self.load_state_dict(state_dict)
            print(f'Loaded model from {prev_rl_checkpoint} with status: {str(load_status)}')

        self.train()

        debug_model_info(self, use_logger=False)

    def sampler_select(self, keep: list):
        if hasattr(self.decoder, 'sampler_select'):
            self.decoder.sampler_select(keep)

    def create_tx_state_encoders(
        self,
        obs_embed_size: int,
        text_embed_size: int,
        prev_action_embed_size: int,
        num_tx_layers: int,
        num_tx_heads: int,
        add_prev_actions: bool,
        add_prev_action_null_token: bool,
        initial_tgt_cache_shape: Tuple[int, int, int],
    ):
        tx_input_size = obs_embed_size

        assert add_prev_action_null_token
        self.last_actions_embed = nn.Embedding(
            self.action_space.n + 2,
            prev_action_embed_size if add_prev_actions else 0,
            padding_idx=self.action_space.n + 1,
        )
        self.last_actions_embed.weight.data.uniform_(-0.01, 0.01)

        state_encoders_params = LLAMAModelArgs(
            dim=obs_embed_size,
            n_layers=num_tx_layers,
            n_heads=num_tx_heads,
            vocab_size=obs_embed_size,
            max_batch_size=initial_tgt_cache_shape[1],
            max_seq_len=initial_tgt_cache_shape[0],
        )

        state_encoders_linear = OrderedDict()
        state_encoders_time = OrderedDict()
        OrderedDict()
        state_encoders = OrderedDict()  # perserve insertion order in py3.6
        if self.multiple_beliefs:  # multiple belief model
            for aux_uuid in self.auxiliary_uuids:
                state_encoders_linear[aux_uuid] = nn.Linear(tx_input_size, self._hidden_size)
                state_encoders_time[aux_uuid] = PositionalEncoder(
                    self._hidden_size, max_len=self.max_steps
                )

                state_encoders[aux_uuid] = LLAMATransformerDecoder(state_encoders_params)
            # create fusion model
            self.fusion_model = self.beliefs_fusion(
                hidden_size=self._hidden_size,
                obs_embed_size=obs_embed_size,
                num_tasks=len(self.auxiliary_uuids),
            )

            self.state_encoders_linear = nn.ModuleDict(state_encoders_linear)
            self.time_encoder = nn.ModuleDict(state_encoders_time)

            self.decoder = nn.ModuleDict(state_encoders)

            self.belief_names = list(self.decoder.keys())

        else:  # single belief model
            if self.use_linear_belief_adaptor:
                self.state_encoders_linear = nn.Linear(tx_input_size, self._hidden_size)
            self.time_encoder = PositionalEncoder(self._hidden_size, max_len=self.max_steps)

            self.decoder = LLAMATransformerDecoder(state_encoders_params)
            self.belief_names = ['single_belief']

        get_logger().info(
            f'there are {len(self.belief_names)} belief models: {self.belief_names}'
        )

    def _recurrent_memory_specification(self):
        return None

    @property
    def is_blind(self) -> bool:
        """True if the model is blind (e.g. neither 'depth' or 'rgb' is an
        input observation type)."""
        return self.visual_encoder.is_blind

    def forward_encoder(self, observations: ObservationType) -> torch.FloatTensor:
        return self.visual_encoder(observations)

    def compute_total_grad_norm(self):
        with torch.no_grad():
            total_norm = 0.0
            for p in self.parameters():
                if p.grad is not None:
                    param_norm = p.grad.detach().data.norm(2)
                    total_norm += param_norm**2
            total_norm = total_norm ** (1.0 / 2)
        return total_norm

    def get_ckpt_path_from_wandb(self, training_run_id, ckptStep):
        import wandb

        api = wandb.Api()
        wandb_entity_name = 'prior-ai2'
        wandb_project_name = 'ilearn_rl'

        on_server = torch.cuda.is_available()
        if on_server:
            output_basedir = '/data/results/online_evaluation'
        else:
            output_basedir = './data/results/online_evaluation'

        run = api.run(f'{wandb_entity_name}/{wandb_project_name}/{training_run_id}')

        eval_run_name = 'OnlineEval' + run.config['exp_name']
        exp_base_dir = os.path.join(output_basedir, eval_run_name)
        ckpt_dir = os.path.join(exp_base_dir, 'ckpts')
        os.makedirs(ckpt_dir, exist_ok=True)

        ckpt_fn = (
            f'{wandb_entity_name}/{wandb_project_name}/ckpt-{training_run_id}-{ckptStep}:latest'
        )
        artifact = api.artifact(ckpt_fn)
        artifact.download(ckpt_dir)
        ckpt_pth = os.path.join(ckpt_dir, 'model.ckpt')

        return ckpt_pth

    def forward(  # type:ignore
        self,
        observations: ObservationType,
        memory: Memory,
        prev_actions: torch.Tensor,
        masks: torch.FloatTensor,
    ) -> Tuple[ActorCriticOutput[DistributionType], Optional[Memory]]:
        """Processes input batched observations to produce new actor and critic
        values. Processes input batched observations (along with prior hidden
        states, previous actions, and masks denoting which recurrent hidden
        states should be masked) and returns an `ActorCriticOutput` object
        containing the model's policy (distribution over actions) and
        evaluation of the current state (value).

        # Parameters
        observations : Batched input observations.
        memory : `Memory` containing the hidden states from initial timepoints.
        prev_actions : Tensor of previous actions taken.
        masks : Masks applied to hidden states. See `RNNStateEncoder`.
        # Returns
        Tuple of the `ActorCriticOutput` and recurrent hidden state.
        """

        # 1.1 use perception model (i.e. encoder) to get observation embeddings
        obs_embeds, text_feats = self.forward_encoder(observations)

        # 1.2 use embedding model to get prev_action embeddings
        if self.last_actions_embed.num_embeddings > self.action_space.n:
            # Instead of first dim as "no_prev_action", we use last dim
            prev_actions_embeds = self.last_actions_embed(
                torch.where(
                    condition=0 != masks.view(*prev_actions.shape),
                    input=prev_actions,
                    other=torch.ones_like(prev_actions) * self.action_space.n,
                )
            )
        else:
            prev_actions_embeds = self.last_actions_embed(prev_actions)

        joint_embeds = obs_embeds + prev_actions_embeds

        # 2. add the object_in_hand embedding
        if self.an_object_is_in_hand_uuid is not None:
            object_in_hand_enc = self.object_in_hand_embed(
                observations[self.an_object_is_in_hand_uuid].squeeze(2)
            )
            joint_embeds = joint_embeds + object_in_hand_enc

        assert not self.multiple_beliefs, 'Multiple beliefs not supported in LLAMA Tx'

        if joint_embeds.shape[0] > 1 or self.time_step_counter >= self.max_steps:
            self.time_step_counter = 0
            # self.done_action_idx = []

        if self.use_linear_belief_adaptor:
            joint_embeds = self.state_encoders_linear(joint_embeds)
        joint_embeds = self.time_encoder(observations[self.time_step_uuid]) + joint_embeds
        x = joint_embeds.permute(1, 0, 2)
        if self.traj_idx_uuid is None:
            mask = None
        elif joint_embeds.shape[0] == 1:
            timesteps = observations[self.time_step_uuid].permute(1, 0)  # bs, nsteps
            epi_start = torch.clamp(self.time_step_counter - timesteps, min=0).expand(
                -1, self.time_step_counter + 1
            )  # bs, 1
            step_range = torch.arange(0, self.time_step_counter + 1).to(device=epi_start.device)

            mask = (epi_start <= step_range).unsqueeze(1).unsqueeze(1)
        else:
            traj_idx: torch.Tensor = observations[self.traj_idx_uuid].permute(1, 0)
            mask = traj_idx[:, :, None] == traj_idx[:, None, :]
            mask = torch.tril(mask)
            mask = mask.unsqueeze(1)  # type: ignore
        y = self.decoder(x, self.time_step_counter, mask)
        beliefs = y.permute(1, 0, 2)
        if joint_embeds.shape[0] == 1:
            self.time_step_counter += 1

        task_weights = None

        # 4. prepare output
        extras = (
            {
                aux_uuid: {
                    'beliefs': beliefs,  # (beliefs_dict[aux_uuid] if self.multiple_beliefs else beliefs),
                    'obs_embeds': obs_embeds,
                    'aux_model': (
                        self.aux_models[aux_uuid] if aux_uuid in self.aux_models else None
                    ),
                }
                for aux_uuid in self.auxiliary_uuids
            }
            if self.auxiliary_uuids is not None
            else {}
        )

        if self.multiple_beliefs:
            extras[MultiAuxTaskNegEntropyLoss.UUID] = task_weights

        total_norm = self.compute_total_grad_norm()
        extras['total_norm'] = torch.Tensor([total_norm])

        if self.critic_type == 'discrete':
            _, stop_grad_logits = self.critic(beliefs.detach())
            extras['stop_grad_logits'] = stop_grad_logits
            extras['loss_func'] = self.critic.loss_fn
            values, full_logits = self.critic(beliefs)
            extras['full_logits'] = full_logits
        else:
            stop_grad_values = self.critic(beliefs.detach())
            extras['stop_grad_values'] = stop_grad_values
            values = self.critic(beliefs)

        if self.critic_type == 'linear':
            with torch.no_grad():
                weight_norm = self.critic.fc.weight.data.norm(2)
                extras['weight_norm'] = torch.Tensor([weight_norm])

                bias_norm = self.critic.fc.bias.data.norm(2)
                extras['bias_norm'] = torch.Tensor([bias_norm])

                if self.critic.fc.weight.grad is not None:
                    weight_grad_norm = self.critic.fc.weight.grad.detach().data.norm(2)
                    extras['weight_grad_norm'] = torch.Tensor([weight_grad_norm])
        else:
            with torch.no_grad():
                weight_norm = self.critic.fc[-1].weight.data.norm(2)
                extras['weight_norm'] = torch.Tensor([weight_norm])

                bias_norm = self.critic.fc[-1].bias.data.norm(2)
                extras['bias_norm'] = torch.Tensor([bias_norm])

                if self.critic.fc[-1].weight.grad is not None:
                    weight_grad_norm = self.critic.fc[-1].weight.grad.detach().data.norm(2)
                    extras['weight_grad_norm'] = torch.Tensor([weight_grad_norm])

        actor_critic_output = ActorCriticOutput(
            distributions=self.actor(beliefs),
            values=values,
            extras=extras,
        )
        return actor_critic_output, memory


class DinoTxGoalEncoder(nn.Module):
    def __init__(
        self,
        observation_spaces: SpaceDict,
        goal_sensor_uuid: str,
        dino_preprocessor_uuid: str,
        manipulation_rgb_dino_preprocessor_uuid: str = None,
        relevant_object_box_uuid: str = None,
        accurate_object_box_uuid: str = None,
        goal_embed_dims: int = 512,
        dino_compressor_hidden_out_dims: Tuple[int, int] = (384, 512),
        combiner_hidden_out_dims: int = 512,
        combiner_layers: int = 3,
        combiner_heads: int = 8,
    ) -> None:
        super().__init__()
        self.goal_uuid = goal_sensor_uuid
        self.dino_uuid = dino_preprocessor_uuid
        self.manip_uuid = manipulation_rgb_dino_preprocessor_uuid
        self.relevant_object_box_uuid = relevant_object_box_uuid
        self.accurate_object_box_uuid = accurate_object_box_uuid
        self.goal_embed_dims = goal_embed_dims
        self.dino_hid_out_dims = dino_compressor_hidden_out_dims
        self.combine_hid_out_dims = combiner_hidden_out_dims

        if goal_sensor_uuid is not None:
            self.goal_space = observation_spaces.spaces[self.goal_uuid]

            text_pt_model = 't5-small'  # "google/flan-t5-small"
            self.text_encoder = T5EncoderModel.from_pretrained(text_pt_model)
            self.text_tokenizer = AutoTokenizer.from_pretrained(text_pt_model)
            self.text_adapter = nn.Sequential(
                nn.Linear(512, self.goal_embed_dims), nn.LayerNorm(self.goal_embed_dims), nn.ReLU()
            )

        self.fusion_token = nn.Parameter(0.1 * torch.rand(self.goal_embed_dims))

        if self.manip_uuid is not None:
            sensor_list = ['raw_navigation_camera', 'raw_manipulation_camera']
        else:
            sensor_list = ['raw_navigation_camera']

        for sensor in sensor_list:
            setattr(
                self,
                f'visual_sensor_token_{sensor}',
                nn.Parameter(0.1 * torch.rand(goal_embed_dims)),
            )

        self.blind = self.dino_uuid not in observation_spaces.spaces
        if not self.blind:
            self.dino_tensor_shape = observation_spaces.spaces[self.dino_uuid].shape
            self.visual_compressor = nn.Sequential(
                nn.Conv2d(self.dino_tensor_shape[-1], self.dino_hid_out_dims[0], 1),
                nn.ReLU(),
                nn.Conv2d(*self.dino_hid_out_dims[0:2], 1),
                nn.ReLU(),
            )

            self.visual_adapter = nn.Sequential(
                nn.Linear(self.dino_hid_out_dims[-1], self.dino_hid_out_dims[-1]),
                nn.LayerNorm(self.dino_hid_out_dims[-1]),
                nn.ReLU(),
            )

            self.fusion_xformer = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    d_model=self.combine_hid_out_dims,
                    nhead=combiner_heads,
                    batch_first=True,
                ),
                num_layers=combiner_layers,
            )

        if relevant_object_box_uuid is not None and accurate_object_box_uuid is not None:
            num_boxes = 2
            num_cameras = 1
            self.len_bounding_boxes = num_boxes * 5 * num_cameras
            self.bbox_pos_encoder = nn.Sequential(
                PositionalEncoder(32),
                nn.Linear(32, self.combine_hid_out_dims),
                nn.LayerNorm(self.combine_hid_out_dims),
                nn.ReLU(),
            )
            self.coord_pos_enc = nn.Embedding(self.len_bounding_boxes, self.combine_hid_out_dims)

    @property
    def is_blind(self):
        return self.blind

    @property
    def output_dims(self):
        if self.blind:
            return self.goal_embed_dims
        else:
            return self.combine_hid_out_dims

    def get_object_type_encoding(
        self, observations: Dict[str, torch.FloatTensor]
    ) -> torch.FloatTensor:
        """Get the object type encoding from input batched observations."""
        return cast(
            torch.FloatTensor,
            self.embed_goal(observations[self.goal_uuid].to(torch.int64)),
        )

    def distribute_target(self, observations):
        max_len = observations[self.goal_uuid].shape[-1]
        goals_tensor = observations[self.goal_uuid].cpu().numpy().astype(np.uint8)
        goals = []
        for g in goals_tensor:
            g = convert_byte_to_string(g, max_len=max_len)
            goals.append(g)
        with torch.no_grad():
            goal_emb = self.text_tokenizer(goals, return_tensors='pt', padding=True).to(
                observations[self.goal_uuid].device
            )
            goal_emb = self.text_encoder(**goal_emb).last_hidden_state
        goal_emb_after_adapter = self.text_adapter(goal_emb)
        return goal_emb_after_adapter

    def encode_bbox(self, observations):
        task_relevant_object_bbox = observations[self.relevant_object_box_uuid]
        nav_accurate_object_bbox = observations[self.accurate_object_box_uuid]
        best_nav_boxes = get_best_of_two_bboxes(task_relevant_object_bbox, nav_accurate_object_bbox)
        B, T, N = best_nav_boxes.shape
        combined_boxes = best_nav_boxes.reshape(B * T, N)
        pos_encoded_boxes = self.bbox_pos_encoder(combined_boxes)
        pos_encoded_boxes = pos_encoded_boxes + self.coord_pos_enc(
            torch.tensor(
                [[i for i in range(self.len_bounding_boxes)]],
                device=pos_encoded_boxes.device,
            ).tile(B * T, 1)
        )
        return pos_encoded_boxes

    def adapt_input(self, observations):
        observations = {**observations}
        dino = observations[self.dino_uuid]
        if self.goal_uuid is not None:
            goal = observations[self.goal_uuid]

        use_agent = False
        nagent = 1

        if len(dino.shape) == 6:
            use_agent = True
            nstep, nsampler, nagent = dino.shape[:3]
        else:
            nstep, nsampler = dino.shape[:2]

        observations[self.dino_uuid] = dino.view(-1, *dino.shape[-3:])
        if self.goal_uuid is not None:
            observations[self.goal_uuid] = goal.view(-1, goal.shape[-1])

        if self.manip_uuid is not None:
            manip = observations[self.manip_uuid]
            observations[self.manip_uuid] = manip.view(-1, *manip.shape[-3:])

        return observations, use_agent, nstep, nsampler, nagent

    @staticmethod
    def adapt_output(x, use_agent, nstep, nsampler, nagent):
        if use_agent:
            return x.view(nstep, nsampler, nagent, -1)
        return x.view(nstep, nsampler * nagent, -1)

    def forward(self, observations):
        observations, use_agent, nstep, nsampler, nagent = self.adapt_input(observations)

        if self.blind:
            return self.embed_goal(observations[self.goal_uuid])

        vis_fit = (
            self.visual_compressor(observations[self.dino_uuid])
            .flatten(start_dim=2)
            .permute(0, 2, 1)
        )
        corresponding_camera_token = getattr(self, f'visual_sensor_token_raw_navigation_camera')

        concatenated_feats = [
            self.fusion_token.view(1, 1, -1).expand(nstep * nsampler, -1, -1),
            self.visual_adapter(vis_fit) + corresponding_camera_token,
        ]

        if self.manip_uuid is not None:
            manip_fit = (
                self.visual_compressor(observations[self.manip_uuid])
                .flatten(start_dim=2)
                .permute(0, 2, 1)
            )
            corresponding_manip_token = getattr(
                self, f'visual_sensor_token_raw_manipulation_camera'
            )
            concatenated_feats.append(self.visual_adapter(manip_fit) + corresponding_manip_token)

        if self.goal_uuid is not None:
            text_feats = self.distribute_target(observations)
            concatenated_feats.append(text_feats)
        else:
            raise NotImplementedError('We currently requires goal sensor to be present')
        if self.relevant_object_box_uuid is not None and self.accurate_object_box_uuid is not None:
            raise NotImplementedError('We currently do not support Bbox observations')
            pos_encoded_boxes = self.encode_bbox(observations)
            concatenated_feats.append(pos_encoded_boxes)
        x = self.fusion_xformer(
            torch.cat(
                concatenated_feats,
                dim=1,
            )
        )
        x = x[:, 0]

        if self.goal_uuid is None:
            return self.adapt_output(x, use_agent, nstep, nsampler, nagent), None
        else:
            return self.adapt_output(x, use_agent, nstep, nsampler, nagent), self.adapt_output(
                text_feats.mean(dim=1), use_agent, nstep, nsampler, nagent
            )


class MLPCriticHead(nn.Module):
    def __init__(self, input_size: int):
        super().__init__()

        self.fc = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

        def init_weights(m):
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight)
                nn.init.constant_(m.bias, 0)

        self.fc.apply(init_weights)

    def forward(self, x):
        return self.fc(x).view(*x.shape[:2], -1)  # [steps, samplers, flattened]


class DiscreteCriticHead(nn.Module):
    def __init__(self, input_size: int, bin_size: int, loss_fn):
        super().__init__()

        self.fc = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Linear(256, bin_size),
        )

        def init_weights(m):
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight)
                nn.init.constant_(m.bias, 0)

        self.fc.apply(init_weights)
        self.loss_fn = loss_fn

    def forward(self, x):
        logits = self.fc(x)
        value = self.loss_fn.transform_from_probs(F.softmax(logits, dim=-1)).view(*x.shape[:2], -1)
        return value, logits
