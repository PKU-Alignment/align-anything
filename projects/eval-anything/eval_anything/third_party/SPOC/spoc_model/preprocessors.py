# Copyright 2024 Allen Institute for AI
# ==============================================================================

import random
from dataclasses import dataclass, field
from typing import List, Tuple

import torch
import torchvision
import torchvision.transforms as T
from torch.distributions.utils import lazy_property
from torch.nn.utils.rnn import pad_sequence
from torchvision.transforms import Compose, Normalize
from transformers import AutoTokenizer

from eval_anything.third_party.SPOC.utils.constants.stretch_initialization_utils import (
    ALL_STRETCH_ACTIONS,
)
from eval_anything.third_party.SPOC.utils.sensor_constant_utils import is_a_visual_sensor
from eval_anything.third_party.SPOC.utils.transformation_util import (
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

    # if size != (224, 384):
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


@dataclass
class PreprocessorConfig:
    image_size: Tuple[int, int] = (224, 384)
    max_steps: int = None
    pad: bool = True
    action_list: List[str] = field(default_factory=lambda: ALL_STRETCH_ACTIONS)
    data_augmentation: bool = True
    augmentation_version: str = 'v2'
    goal_sensor_uuid: str = 'goals'
    model_version: str = ''
    text_encoder_context_length: int = None


class Preprocessor:
    def __init__(self, cfg: PreprocessorConfig, device: torch.device):
        self.cfg = cfg
        self.device = device
        self.action2idx = {action: i for i, action in enumerate(self.cfg.action_list)}
        self.action2idx[''] = len(
            self.cfg.action_list
        )  # for start of sequence token in last_actions

    @lazy_property
    def image_preprocessor(self):
        return tensor_image_preprocessor(
            size=self.cfg.image_size,
            data_augmentation=self.cfg.data_augmentation,
            augmentation_version=self.cfg.augmentation_version,
        )

    @lazy_property
    def text_preprocessor(self):
        return AutoTokenizer.from_pretrained('t5-small')

    @property
    def num_actions(self):
        return len(self.action2idx)

    def process_frames(self, batch, sensor_key):
        frame_processor = self.get_frame_processor(sensor_key)
        frames = list(map(frame_processor, batch))
        if self.cfg.pad:
            return pad_sequence(frames, batch_first=True, padding_value=0)

        return frames

    def get_frame_processor(self, sensor_key):
        def frame_processor(sample):
            frames = sample[sensor_key][: self.cfg.max_steps].to(self.device)
            frames = frames.permute(0, 3, 1, 2)

            res = self.image_preprocessor(frames)

            return res

        return frame_processor

    def process_actions(self, batch):
        action_processor = self.get_action_processor()
        actions = list(map(action_processor, batch))
        if self.cfg.pad:
            return pad_sequence(actions, batch_first=True, padding_value=-1)

        return actions

    def get_action_processor(self):
        def action_processor(sample):
            actions = sample['actions'][: self.cfg.max_steps]
            actions = torch.tensor(
                [self.action2idx[action] for action in actions], dtype=torch.int64
            ).to(self.device)
            return actions

        return action_processor

    def process_last_actions(self, batch):
        last_actions_processor = self.get_last_actions_processor()
        last_actions = list(map(last_actions_processor, batch))
        if self.cfg.pad:
            return pad_sequence(last_actions, batch_first=True, padding_value=len(self.action2idx))

        return last_actions

    def get_last_actions_processor(self):
        def last_actions_processor(sample):
            last_actions = sample['last_actions'][: self.cfg.max_steps]
            last_actions = torch.tensor(
                [self.action2idx[action] for action in last_actions], dtype=torch.int64
            ).to(self.device)
            return last_actions

        return last_actions_processor

    def process_goals(self, batch):
        goal_spec = self.text_preprocessor(
            [sample['goal'] for sample in batch],
            return_tensors='pt',
            padding=True,
        )
        return {k: v.to(self.device) for k, v in goal_spec.items()}

    def process_visibility(self, batch):
        visibility = [torch.tensor(sample['visibility']) for sample in batch]
        if self.cfg.pad:
            return pad_sequence(visibility, batch_first=True, padding_value=-1).to(self.device)

        return visibility

    def process_rooms_seen(self, batch, key='rooms_seen'):
        rooms_seen = [torch.tensor(sample[key]) for sample in batch]
        if self.cfg.pad:
            return pad_sequence(rooms_seen, batch_first=True, padding_value=19).to(self.device)

        return rooms_seen

    def process_room_current_seen(self, batch, key='room_current_seen'):
        room_current_seen = [torch.tensor(sample[key], dtype=torch.int64) for sample in batch]
        if self.cfg.pad:
            return pad_sequence(room_current_seen, batch_first=True, padding_value=2).to(
                self.device
            )

        return room_current_seen

    def process_time_ids(self, batch):
        time_ids = [torch.tensor(sample['time_ids'][: self.cfg.max_steps]) for sample in batch]
        if self.cfg.pad:
            return pad_sequence(time_ids, batch_first=True, padding_value=-1).to(self.device)

        return time_ids

    def process_objinhand(self, batch):
        # torch.tensor(sample["an_object_is_in_hand"][: self.cfg.max_steps]).long()
        obj_in_hand = [
            sample['an_object_is_in_hand'][: self.cfg.max_steps].clone().detach().long()
            for sample in batch
        ]
        if self.cfg.pad:
            return pad_sequence(obj_in_hand, batch_first=True, padding_value=2).to(self.device)

        return obj_in_hand

    def process_arm_proprioceptive(self, batch):
        arm_proprioceptive = [
            torch.tensor(sample['relative_arm_location_metadata'][: self.cfg.max_steps]).float()
            for sample in batch
        ]
        if self.cfg.pad:
            return pad_sequence(arm_proprioceptive, batch_first=True, padding_value=-1).to(
                self.device
            )

        return arm_proprioceptive

    def process_task_relevant_bbox(self, batch, sensor):
        task_relevant_object_bbox = [torch.tensor(sample[sensor]).float() for sample in batch]
        if self.cfg.pad:
            return pad_sequence(task_relevant_object_bbox, batch_first=True, padding_value=-1).to(
                self.device
            )

    def create_padding_mask(self, lengths, max_length):
        # Create a range tensor with the shape (1,max_length)
        range_tensor = torch.arange(max_length, device=self.device).unsqueeze(0)
        return range_tensor >= lengths.unsqueeze(1)

    def process(self, batch):
        if len(batch) == 0:
            return None

        batch = [sample['observations'] for sample in batch]

        batch_keys = list(batch[0].keys())
        output = dict()

        for sensor in batch_keys:
            if is_a_visual_sensor(sensor):
                output[sensor] = self.process_frames(batch, sensor_key=sensor)
            elif sensor == 'an_object_is_in_hand':
                output[sensor] = self.process_objinhand(batch)
            elif sensor == 'relative_arm_location_metadata':
                output[sensor] = self.process_arm_proprioceptive(batch)
            elif sensor == 'actions':
                output['actions'] = self.process_actions(batch)
            elif sensor == 'last_actions':
                output['last_actions'] = self.process_last_actions(batch)
            elif sensor == 'goal':
                output[self.cfg.goal_sensor_uuid] = self.process_goals(batch)
            elif sensor == 'time_ids':
                output['time_ids'] = self.process_time_ids(batch)
            elif sensor == 'visibility':
                output['visibility'] = self.process_visibility(batch)
            elif sensor in ['rooms_seen', 'rooms_seen_output']:
                output[sensor] = self.process_rooms_seen(batch, key=sensor)
            elif sensor in ['room_current_seen', 'room_current_seen_output']:
                output[sensor] = self.process_room_current_seen(batch, key=sensor)
            elif sensor in [
                'nav_task_relevant_object_bbox',
                'manip_task_relevant_object_bbox',
                'nav_accurate_object_bbox',
                'manip_accurate_object_bbox',
            ]:
                output[sensor] = self.process_task_relevant_bbox(batch, sensor)
            else:
                if sensor not in ['initial_agent_location', 'templated_task_type']:
                    raise NotImplementedError(f'Sensor {sensor} not implemented')

        if 'actions' in batch_keys:
            key_to_look_at = 'actions'
        else:
            key_to_look_at = random.choice([k for k in batch_keys if is_a_visual_sensor(k)])

        output['lengths'] = torch.tensor(
            [len(sample[key_to_look_at]) for sample in batch], dtype=torch.int32
        ).to(self.device)

        if self.cfg.pad:
            output['padding_mask'] = self.create_padding_mask(
                output['lengths'], output[key_to_look_at].shape[1]
            )

        return output
