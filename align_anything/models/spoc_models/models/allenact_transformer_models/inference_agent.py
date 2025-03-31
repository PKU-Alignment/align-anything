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
import base64
import io
import json
import os
from typing import Optional, Tuple, cast

import attr
import numpy as np
import torch
import torchvision
import torchvision.transforms as T
from allenact.base_abstractions.misc import (
    ActorCriticOutput,
    DistributionType,
    Memory,
    ObservationType,
)
from allenact.utils import spaces_utils as su
from allenact.utils.inference import InferenceAgent
from allenact.utils.tensor_utils import batch_observations
from PIL import Image
from torchvision.transforms import Compose, Normalize

from align_anything.utils.spoc_utils.constants.stretch_initialization_utils import (
    ALL_STRETCH_ACTIONS,
)
from align_anything.utils.spoc_utils.string_utils import convert_string_to_byte
from align_anything.utils.spoc_utils.transformation_util import (
    get_full_transformation_list,
    sample_a_specific_transform,
)


def tensor_image_preprocessor(
    size=(224, 384),
    data_augmentation=False,
    specific=False,
    augmentation_version='v2',
    mean=(0.48145466, 0.4578275, 0.40821073),
    std=(0.26862954, 0.26130258, 0.27577711),
):
    def convert_to_float(tensor):
        return tensor.float() / 255.0

    list_of_transformations = []

    if size != (224, 384):
        list_of_transformations += [
            torchvision.transforms.Resize(
                size,
                interpolation=T.InterpolationMode('bicubic'),
                max_size=None,
                antialias=True,
            )
        ]

    if data_augmentation:
        data_aug_transforms = get_full_transformation_list(size=size, version=augmentation_version)
        if specific:
            data_aug_transforms = sample_a_specific_transform(
                Compose(data_aug_transforms)
            ).transforms

        list_of_transformations += data_aug_transforms

    list_of_transformations += [
        torchvision.transforms.Lambda(convert_to_float),
        Normalize(mean=mean, std=std),
    ]
    return Compose(list_of_transformations)


def encode_image(image_array):
    # Convert the NumPy array to a PIL image
    image = Image.fromarray(image_array.astype('uint8'))

    # Save the image to a bytes buffer
    buffered = io.BytesIO()
    image.save(buffered, format='PNG')

    # Encode the bytes buffer to base64
    return base64.b64encode(buffered.getvalue()).decode('utf-8')


class InferenceAgentVIDA(InferenceAgent):
    img_encoder_rgb_mean = attr.ib()
    img_encoder_rgb_std = attr.ib()
    greedy_sampling: bool = attr.ib()
    test_augmentation: bool = attr.ib()
    augmentations = attr.ib()
    num_evaluated_traj = 0

    @classmethod
    def build_agent(
        cls,
        exp_config_type,
        params,
        device,
        img_encoder_rgb_mean,
        img_encoder_rgb_std,
        greedy_sampling,
        test_augmentation,
        ckpt_path,
    ):
        exp_config = exp_config_type(params=params)
        agent = cls.from_experiment_config(
            exp_config=exp_config,
            mode='test',
            device=device,
        )
        agent.img_encoder_rgb_mean = img_encoder_rgb_mean
        agent.img_encoder_rgb_std = img_encoder_rgb_std
        agent.greedy_sampling = greedy_sampling
        agent.test_augmentation = test_augmentation
        agent.augmentations = tensor_image_preprocessor(
            size=(256, 256),
            data_augmentation=True,
            augmentation_version='v2',
            mean=img_encoder_rgb_mean,
            std=img_encoder_rgb_std,
        )
        model = torch.load(
            ckpt_path, map_location='cpu' if not torch.cuda.is_available() else 'cuda'
        )
        if 'model_state_dict' in model.keys():
            model = model['model_state_dict']
        else:
            model = model
        agent.actor_critic.load_state_dict(
            model,
            # torch.load(ckpt_path, map_location="cpu" if not torch.cuda.is_available() else "cuda")[
            # "model_state_dict"
            # ],
            strict=False,
        )
        agent.steps_before_rollout_refresh = 10000

        agent.reset()
        return agent

    def reset(self):
        if self.has_initialized:
            self.rollout_storage.after_updates()
        self.steps_taken_in_task = 0
        self.num_evaluated_traj += 1
        self.memory = None

    def normalize_img(self, frame):
        frame -= (
            torch.from_numpy(np.array(self.img_encoder_rgb_mean))
            .to(device=self.device)
            .float()
            .view(1, 1, 1, 3)
        )
        frame /= (
            torch.from_numpy(np.array(self.img_encoder_rgb_std))
            .to(device=self.device)
            .float()
            .view(1, 1, 1, 3)
        )
        return frame

    def get_action_list(self):
        if os.getenv('ACTION_DICT') is not None:
            assert os.path.exists(os.getenv('ACTION_DICT'))
            return list(json.load(open(os.getenv('ACTION_DICT'))).keys())
        else:
            return ALL_STRETCH_ACTIONS

    def get_test_augmentation(self, frame):
        frame = self.augmentations(Image.fromarray(frame))
        return frame

    def get_action(self, frame, goal_spec):
        observations = {
            'rgb_raw': frame['raw_navigation_camera'],
            'natural_language_spec': convert_string_to_byte(goal_spec, 1000),
            'time_step': self.steps_taken_in_task,
            'traj_index': self.num_evaluated_traj,
        }
        if 'raw_manipulation_camera' in frame.keys():
            observations['manipulation_rgb_raw'] = frame['raw_manipulation_camera']
        if 'an_object_is_in_hand' in frame.keys():
            observations['an_object_is_in_hand'] = frame['an_object_is_in_hand']
        if 'relative_arm_location_metadata' in frame.keys():
            full_pose = frame['relative_arm_location_metadata']
            full_pose[-1] = full_pose[-1] * np.pi / 180
            full_pose[-1] = (full_pose[-1] + np.pi) % (2 * np.pi) - np.pi
            observations['relative_arm_location_metadata'] = full_pose
        if 'nav_accurate_object_bbox' in frame.keys():
            observations['nav_accurate_object_bbox'] = frame['nav_accurate_object_bbox']
        if 'nav_task_relevant_object_bbox' in frame.keys():
            observations['nav_task_relevant_object_bbox'] = frame['nav_task_relevant_object_bbox']
        return self.act(observations, goal_spec)

    def act(self, observations: ObservationType, goal_spec):
        obs_batch = batch_observations([observations], device=self.device)
        if self.sensor_preprocessor_graph is not None:
            if 'graph' in self.sensor_preprocessor_graph.compute_order:
                self.sensor_preprocessor_graph.compute_order.pop(
                    self.sensor_preprocessor_graph.compute_order.index('graph')
                )
            obs_batch = self.sensor_preprocessor_graph.get_observations(obs_batch)
        if 'rgb_dino_vit' in obs_batch.keys():
            obs_batch['rgb_dino_vit'] = (
                obs_batch['rgb_dino_vit'].flatten(start_dim=2).permute(0, 2, 1)
            )
        if 'graph' in obs_batch.keys():
            obs_batch.pop('graph')

        if self.steps_taken_in_task == 0:
            self.has_initialized = True
            self.rollout_storage.initialize(
                observations=obs_batch,
                num_samplers=1,
                recurrent_memory_specification=self.actor_critic.recurrent_memory_specification,
                action_space=self.actor_critic.action_space,
            )
            self.rollout_storage.after_updates()
        else:
            dummy_val = torch.zeros((1, 1), device=self.device)  # Unused dummy value
            self.rollout_storage.add(
                observations=obs_batch,
                memory=self.memory,
                actions=self.last_action_flat[0],
                action_log_probs=dummy_val,
                value_preds=dummy_val,
                rewards=dummy_val,
                costs=dummy_val,
                c_value_preds=dummy_val,
                masks=torch.ones(
                    (1, 1), device=self.device
                ),  # Always == 1 as we're in a single task until `reset`
            )

        agent_input = self.rollout_storage.agent_input_for_next_step()

        actor_critic_output, self.memory = cast(
            Tuple[ActorCriticOutput[DistributionType], Optional[Memory]],
            self.actor_critic(**agent_input),
        )

        action = actor_critic_output.distributions.sample()

        action_greedy = actor_critic_output.distributions.mode()

        # NOTE: Last action flat is always stochastic
        self.last_action_flat = su.flatten(self.actor_critic.action_space, action)

        self.steps_taken_in_task += 1

        if self.steps_taken_in_task % self.steps_before_rollout_refresh == 0:
            self.rollout_storage.after_updates()

        if self.greedy_sampling:
            action_str = self.get_action_list()[
                su.action_list(self.actor_critic.action_space, action_greedy)[0]
            ]
        else:
            action_str = self.get_action_list()[
                su.action_list(self.actor_critic.action_space, self.last_action_flat)[0]
            ]

        return action_str, actor_critic_output.distributions.probs[0][0]
