# Copyright 2024 PKU-Alignment Team. All Rights Reserved.
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


import json
import os
import platform
import random
import traceback
from copy import deepcopy
from typing import List, Optional, Sequence

import h5py
import numpy as np
from torch.utils.data import Dataset
from torchvision.io import read_video
from tqdm import tqdm

from align_anything.utils.spoc_utils.sensor_constant_utils import is_a_visual_sensor
from align_anything.utils.spoc_utils.string_utils import (
    convert_byte_to_string,
    json_templated_spec_to_dict,
    json_templated_to_NL_spec,
)


class ChoresDataReader:
    def __init__(self, data_dir, subset, max_samples=None, seed=123):
        self.data_dir = data_dir
        self.subset = subset
        self.max_samples = max_samples
        self.seed = seed
        self.house_id_to_sub_house_id_json = os.path.join(
            data_dir, f'house_id_to_sub_house_id_{self.subset}.json'
        )

        self.dataset_version = self.data_dir.split('/')[-1].split('_')[-1]

    def load_samples(self):
        # select house files based on the process id
        with open(self.house_id_to_sub_house_id_json) as f:
            house_id_to_sub_house_id = json.load(f)

        house_ids = sorted(list(house_id_to_sub_house_id.keys()))
        assert len(house_ids) > 0, f"{self.data_dir}/{self.subset} doesn't exist or has no houses"

        random.seed(self.seed)
        random.shuffle(house_ids)

        samples = []
        for house_id in tqdm(house_ids):
            house_dir = os.path.join(self.data_dir, self.subset, house_id)
            for sub_house_id in house_id_to_sub_house_id[house_id]:
                sample_id = f'house={house_id},sub_house_id={sub_house_id}'
                sensors_path = os.path.join(house_dir, 'hdf5_sensors.hdf5')
                nav_cam_video = os.path.join(
                    house_dir, f'raw_navigation_camera__{sub_house_id}.mp4'
                )
                manip_cam_video = nav_cam_video.replace('navigation', 'manipulation')
                samples.append(
                    dict(
                        sample_id=sample_id,
                        house_id=house_id,
                        sub_house_id=sub_house_id,
                        raw_navigation_camera=nav_cam_video,
                        raw_manipulation_camera=manip_cam_video,
                        sensors_path=sensors_path,
                    )
                )
        random.seed(self.seed)
        random.shuffle(samples)
        return samples[: self.max_samples]

    def read_sensors(
        self, sensors_path, sub_house_id, additional_sensor_keys: Optional[List[str]] = None
    ):
        if additional_sensor_keys is None:
            additional_sensor_keys = []

        with h5py.File(sensors_path, 'r') as f:
            grp = f[sub_house_id]
            sensor_keys = [
                'last_action_str',
                'initial_agent_location',
                'templated_task_spec',
            ] + additional_sensor_keys

            sensors = dict()
            task_dict = convert_byte_to_string(grp['templated_task_spec'][0], None)
            task_dict = json_templated_spec_to_dict(task_dict)

            for k in sensor_keys:
                if k == 'initial_agent_location':
                    sensors[k] = grp['last_agent_location'][0]
                elif k == 'last_action_str':
                    sensors[k] = [convert_byte_to_string(row, None) for row in grp[k]]
                elif k == 'last_actions':
                    # last_actions are created from last_action_str which is always returned
                    continue
                elif k == 'templated_task_spec':
                    sensors[k] = convert_byte_to_string(grp[k][0], None)
                elif k == 'visible_target_4m_count':
                    sensors[k] = grp[k][:, 0]
                elif k in ['rooms_seen', 'room_current_seen']:
                    sensors[k] = grp[k][:-1]
                    sensors[f'{k}_output'] = grp[k][1:]
                elif k == 'an_object_is_in_hand':
                    if k in grp:
                        sensors[k] = grp[k][:, 0]
                    else:
                        sensors[k] = np.zeros(len(sensors['last_action_str']))
                    assert np.all(sensors[k] <= 1)
                elif k == 'relative_arm_location_metadata':
                    sensors[k] = grp[k][:]
                elif k in [
                    'nav_task_relevant_object_bbox',
                    'manip_task_relevant_object_bbox',
                    'nav_accurate_object_bbox',
                    'manip_accurate_object_bbox',
                ]:
                    if task_dict['task_type'] == 'RoomVisit':
                        seq_len = len(sensors['last_action_str'])
                        sensors[k] = np.zeros((seq_len, 10))
                        sensors[k][:, :4] = 1000
                        sensors[k][:, 5:9] = 1000
                        continue
                    num_boxes = grp[k]['min_cols'].shape[1]

                    oids = eval(convert_byte_to_string(grp[k]['oids_as_bytes'][0]))
                    assert len(oids) == num_boxes, "Number of oids and boxes don't match"

                    tgt_1_ids = []
                    tgt_2_ids = []

                    if 'broad_synset_to_object_ids' in task_dict:
                        tgt_1_ids = [
                            val for val in task_dict['broad_synset_to_object_ids'].values()
                        ]
                        tgt_1_ids = sum(tgt_1_ids, [])

                    def parse_biggest_bbox(object_indices):
                        object_indices = sorted(object_indices)
                        if (
                            len(object_indices) == 0
                        ):  # both bbox_1 and bbox_2 need to have a default value
                            res = np.zeros((len(grp[k]['min_cols']), 5))
                            res[:, :4] = 1000  # res[:, 4] = 0 by default
                            return res
                        x1 = grp[k]['min_cols'][:, object_indices].astype(int).astype(np.float32)
                        y1 = grp[k]['min_rows'][:, object_indices].astype(int).astype(np.float32)
                        x2 = grp[k]['max_cols'][:, object_indices].astype(int).astype(np.float32)
                        y2 = grp[k]['max_rows'][:, object_indices].astype(int).astype(np.float32)
                        if np.any(x1 > x2):
                            x1, x2 = x2, x1
                        if np.any(y1 > y2):
                            y1, y2 = y2, y1
                        area = (y2 - y1) * (x2 - x1)
                        largest_area_oids = np.argmax(area, axis=1)
                        time_ids = np.arange(len(x1))
                        bboxes = np.stack(
                            [
                                x1[time_ids, largest_area_oids],
                                y1[time_ids, largest_area_oids],
                                x2[time_ids, largest_area_oids],
                                y2[time_ids, largest_area_oids],
                                area[time_ids, largest_area_oids],
                            ],
                            axis=1,
                        )
                        bboxes[bboxes == -1] = 1000
                        return bboxes

                    bbox_1 = parse_biggest_bbox([oids.index(oid) for oid in tgt_1_ids])
                    bbox_2 = parse_biggest_bbox([oids.index(oid) for oid in tgt_2_ids])
                    bboxes_combined = np.concatenate([bbox_1, bbox_2], axis=1)
                    bbox_to_return = bboxes_combined

                    sensors[k] = bbox_to_return

                else:
                    raise NotImplementedError(f'Sensor {k} not implemented')

        return sensors

    def load_video(self, video_path):
        return read_video(filename=video_path, pts_unit='sec')[0]


class ChoresDataset(Dataset):
    def __init__(
        self,
        data_dir,
        subset,
        max_samples=None,
        load_frames=True,
        sliding_window=None,
        input_sensors=('raw_navigation_camera',),
        reduce_action_redundancy=False,
    ):
        self.data_dir = data_dir
        self.reader = ChoresDataReader(data_dir, subset, max_samples)
        self.load_frames = load_frames
        self.sliding_window = sliding_window
        self.samples = self.reader.load_samples()
        self.visual_sensors = [x for x in input_sensors if is_a_visual_sensor(x)]
        self.non_visual_sensors = [x for x in input_sensors if not is_a_visual_sensor(x)]
        self.reduce_action_redundancy = reduce_action_redundancy
        self.add_boxes_to_img = False

        assert (
            not self.reduce_action_redundancy
        ) or subset == 'train', 'Reducing action redundancy is currently only supported for train.'

        self.prob_sample_last_steps = 0

    def __len__(self):
        return len(self.samples)

    def set_prob_sample_last_steps(self, prob):
        self.prob_sample_last_steps = prob

    def select_window_slice(self, len_of_time_inds: int, start_idx=None, sliding_window=None):
        if sliding_window is None:
            sliding_window = self.sliding_window

        if sliding_window is None or len_of_time_inds <= sliding_window:
            return slice(0, len_of_time_inds)

        if start_idx is None:
            if random.random() < self.prob_sample_last_steps:
                start_idx = len_of_time_inds - sliding_window
            else:
                start_idx = random.randint(0, len_of_time_inds - sliding_window)

        return slice(start_idx, start_idx + sliding_window)

    def __getitem__(self, i):
        sample = deepcopy(self.samples[i])

        sensors = self.reader.read_sensors(
            sample['sensors_path'], sample['sub_house_id'], self.non_visual_sensors
        )

        # throw out the first action (null action)
        actions = sensors['last_action_str'][1:]
        original_actions_len = len(actions)

        select_indices = self.select_window_slice(original_actions_len)

        time_ids = np.arange(original_actions_len)[select_indices]
        actions = np.array(actions)[select_indices]

        # Randomly shift the time indices for training
        if self.reader.subset == 'train':
            time_ids = time_ids + random.randint(0, max(1000 - time_ids.shape[0], 0))

        input_sensors = {}
        output_sensors = {}
        if self.load_frames:
            # throw out the last frame cause it doesn't have an action

            for sensor_name in self.visual_sensors:
                frames = self.reader.load_video(sample[sensor_name])[:-1]

                original_frames_len = len(frames)

                frames = frames[select_indices]
                msg = (
                    'frames and actions do not match for sample id '
                    + sample['sample_id']
                    + sample[sensor_name]
                )

                try:
                    assert original_frames_len == original_actions_len, print(
                        msg,
                        'original_frame',
                        original_frames_len,
                        'original actions',
                        original_actions_len,
                    )  # KE: This is really important please do not remove
                except Exception as e:
                    print('In Lengths not equal')
                    print(str(e))
                    print(traceback.format_exc())
                    print(msg)
                    return None
                input_sensors[sensor_name] = frames
                # If dimensions of the video are off just throw out this sample and print a warning
                if frames.shape[1] != 224 or frames.shape[2] != 384:
                    print('Video dimensions are not 224x384, throwing out this sample')
                    print('instead they are', frames.shape)
                    print('sample path ', sample[sensor_name])
                    return None

            for sensor_name in self.non_visual_sensors:
                if sensor_name in ['rooms_seen', 'room_current_seen']:
                    input_sensors[sensor_name] = sensors[sensor_name][select_indices]
                    output_sensors[f'{sensor_name}_output'] = sensors[f'{sensor_name}_output'][
                        select_indices
                    ]
                elif sensor_name == 'last_actions':
                    input_sensors[sensor_name] = np.array(sensors['last_action_str'][:-1])[
                        select_indices
                    ]

                else:
                    # The sensors are created after the action is taken. So they have to
                    # include an additional step in the beginning similar to the first
                    # action, as a result we should only take the first n-1 sensors, ask KE for more details
                    input_sensors[sensor_name] = sensors[sensor_name][:-1][select_indices]

                    if (
                        sensor_name
                        in ['nav_task_relevant_object_bbox', 'manip_task_relevant_object_bbox']
                        and self.add_boxes_to_img
                    ):
                        raise NotImplementedError(
                            'not working this needs to be changed to cover both boxes'
                        )
        sample['observations'] = dict(
            **input_sensors,
            **output_sensors,
            actions=actions,
            goal=json_templated_to_NL_spec(sensors['templated_task_spec']),  # [0]),
            time_ids=time_ids,
            initial_agent_location=sensors['initial_agent_location'],
            templated_task_type=sensors['templated_task_spec'],
        )
        sample['prob_sample_last_steps'] = self.prob_sample_last_steps
        return sample


class ChoresMultitaskDataset(Dataset):
    def __init__(self, base_data_dir: str, dataset_names: Sequence[str], **kwargs):
        super().__init__()
        self.base_data_dir = base_data_dir
        self.dataset_names = dataset_names

        self.datasets = []
        for name in self.dataset_names:
            print(f'Loading dataset: {name}')
            data_dir = os.path.join(base_data_dir, name)
            self.datasets.append(ChoresDataset(data_dir, **kwargs))
        self.max_size = max(len(d) for d in self.datasets)

        # properties for last steps preference
        self.curr_prob_sample_last_steps = 0
        self.prob_decay_size = 0

    def __len__(self):
        return self.max_size * len(self.datasets)

    def set_prob_sample_last_steps(self, prob):
        for d in self.datasets:
            d.set_prob_sample_last_steps(prob)

    def init_prob_sample_last_steps(
        self, init_prob, final_prob, num_workers=4, num_gpu_per_node=1, num_node=1
    ):
        if num_gpu_per_node == 0:
            assert platform.system() == 'Darwin'
            num_gpu_per_node = 1

        self.curr_prob_sample_last_steps = init_prob
        self.prob_decay_size = (init_prob - final_prob) / (
            len(self) / (num_workers * num_gpu_per_node * num_node)
        )

    def set_up_probabilty_to_sample_last_steps(self):
        self.curr_prob_sample_last_steps -= self.prob_decay_size
        self.set_prob_sample_last_steps(self.curr_prob_sample_last_steps)

    def __getitem__(self, index):
        try:
            # return sample in order D0[0],D1[0],D0[1],D1[1]
            # Choose the dataset based on the index
            dataset_index = index % len(self.datasets)
            dataset = self.datasets[dataset_index]

            # Choose the sample from the selected dataset
            sample_index = index // len(self.datasets)
            sample_index = sample_index % len(dataset)  # wrap around if out of range
            res = dataset[sample_index]
        except (KeyboardInterrupt, SystemExit):
            raise
        except:
            res = None
        self.set_up_probabilty_to_sample_last_steps()
        return res
