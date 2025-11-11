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

import json
import os
import sys
import traceback
from collections import namedtuple
from datetime import datetime
from itertools import chain
from typing import Any, Dict, List, Optional

import ai2thor.platform
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from eval_anything.dataloader.tv2act_dataloader import TV2ACTDataLoader
from eval_anything.pipeline.base_benchmark import BaseBenchmark
from eval_anything.third_party.SPOC.environment.stretch_controller import StretchController
from eval_anything.third_party.SPOC.spoc_model.agent import AbstractAgent
from eval_anything.third_party.SPOC.tasks.abstract_task import AbstractSafeTask
from eval_anything.third_party.SPOC.tasks.multi_task_eval_sampler import MultiTaskSampler
from eval_anything.third_party.SPOC.tasks.task_specs import TaskSpecList
from eval_anything.third_party.SPOC.utils.constants.stretch_initialization_utils import (
    STRETCH_ENV_ARGS,
)
from eval_anything.third_party.SPOC.utils.data_generation_utils.mp4_utils import save_frames_to_mp4
from eval_anything.third_party.SPOC.utils.online_evaluation_types_and_utils import (
    calc_trajectory_room_visitation,
)
from eval_anything.third_party.SPOC.utils.task_datagen_utils import get_core_task_args
from eval_anything.third_party.SPOC.utils.type_utils import THORActions
from eval_anything.third_party.SPOC.utils.visualization_utils import (
    VideoLogging,
    get_top_down_frame,
)
from eval_anything.utils.data_type import InferenceInput, InferenceOutput
from eval_anything.utils.register import BenchmarkRegistry


@BenchmarkRegistry.register('text_vision_to_action')
class TV2ACTBenchmark(BaseBenchmark):
    def __init__(self, eval_cfgs, model_cfgs, infer_cfgs, output_path, cache_manager, logger):
        self.max_eps_len = infer_cfgs.max_eps_len
        self.gpu_devices = infer_cfgs.gpu_devices
        self.gpu_device = infer_cfgs.gpu_device
        self.sampling = infer_cfgs.sampling
        self.task_type = infer_cfgs.task_type
        self.skip_done = infer_cfgs.skip_done
        self.logging_sensor = VideoLogging()
        self.output_path = output_path
        self.input_sensors = model_cfgs.model_input_sensors
        self.pre_defined_max_steps = self.max_eps_len
        self._task_sampler: Optional[MultiTaskSampler] = None

    def init_dataloader(self, eval_cfgs: namedtuple = None, data_cfgs: namedtuple = None):

        dataloader = TV2ACTDataLoader(data_cfgs)
        self.input_data, self.tasksets = dataloader.load_task_dataset()
        self.house_assets = dataloader.load_house_assets()
        return self.input_data

    def save_benchmark_details(
        self, save_path: str, benchmark_name: str, results_lst: List, details_lst: List
    ):
        """Save evaluation result and config file of single benchmark.
        Args:
            save_path (str): save path
            inputs (dict[str, List[InferenceInput]]): evaluation inputs
            results (dict[str, List[EvaluationResult]]): evaluation results
        """
        output_dir = os.path.join(save_path, benchmark_name)
        os.makedirs(output_dir, exist_ok=True)
        datetime_str = datetime.now().strftime('%Y%m%d_%H%M%S')
        video_dir = os.path.join(output_dir, datetime_str)
        os.makedirs(video_dir, exist_ok=True)
        with open(
            os.path.join(video_dir, f'details_{self.task_type}.jsonl'), 'a', encoding='utf-8'
        ) as f:
            for result in results_lst:
                f.write(json.dumps(result, ensure_ascii=False) + '\n')

        for detail in tqdm(details_lst):
            task_dir = os.path.join(video_dir, detail['name'])
            os.makedirs(task_dir, exist_ok=True)
            frist_view_mp4_name = 'primary' + '.mp4'
            bev_mp4_name = 'bev' + '.mp4'
            navigation_mp4_name = 'navigation' + '.mp4'
            manipulation_mp4_name = 'manipulation' + '.mp4'
            third_person_mp4_name = 'third_person' + '.mp4'
            frist_view_video_path_to_send = os.path.join(task_dir, frist_view_mp4_name)
            bew_video_path_to_send = os.path.join(task_dir, bev_mp4_name)
            navigation_video_path_to_send = os.path.join(task_dir, navigation_mp4_name)
            manipulation_video_path_to_send = os.path.join(task_dir, manipulation_mp4_name)
            third_person_video_path_to_send = os.path.join(task_dir, third_person_mp4_name)
            save_frames_to_mp4(
                frames=detail['all_video_frames'], file_path=frist_view_video_path_to_send, fps=5
            )
            save_frames_to_mp4(
                frames=detail['top_down_video_frames'], file_path=bew_video_path_to_send, fps=5
            )
            save_frames_to_mp4(
                frames=detail['navigation_video_frames'],
                file_path=navigation_video_path_to_send,
                fps=5,
            )
            save_frames_to_mp4(
                frames=detail['manipulation_video_frames'],
                file_path=manipulation_video_path_to_send,
                fps=5,
            )
            save_frames_to_mp4(
                frames=detail['third_person_video_frames'],
                file_path=third_person_video_path_to_send,
                fps=5,
            )

            topdown_view_path = os.path.join(task_dir, detail['name'] + '_topdown.png')
            plt.imsave(fname=topdown_view_path, arr=detail['top_down_frame'])
        print(f'Saved video to {save_path}')

    def calculate_overall_metrics(self, result):
        avg_succ = 0
        avg_cost_danger = 0
        avg_cost_corner = 0
        avg_cost_blind = 0
        avg_cost_fragile = 0
        avg_cost_critical = 0
        avg_eps_len = 0
        for res in result:
            avg_succ += res['metrics']['success']
            avg_cost_danger += res['metrics']['cost_danger']
            avg_cost_corner += res['metrics']['cost_corner']
            avg_cost_blind += res['metrics']['cost_blind']
            avg_cost_fragile += res['metrics']['cost_fragile']
            avg_cost_critical += res['metrics']['cost_critical']
            # avg_robot_cost += res['metrics']['cost_robot']
            # avg_object_cost += res['metrics']['cost_object']
            avg_eps_len += res['metrics']['eps_len']
        avg_succ /= len(result)
        # avg_robot_cost /= len(result)
        # avg_object_cost /= len(result)
        avg_cost_danger /= len(result)
        avg_cost_corner /= len(result)
        avg_cost_blind /= len(result)
        avg_cost_fragile /= len(result)
        avg_cost_critical /= len(result)
        avg_eps_len /= len(result)
        return dict(
            avg_succ=avg_succ,
            avg_cost_danger=avg_cost_danger,
            avg_cost_corner=avg_cost_corner,
            avg_cost_blind=avg_cost_blind,
            avg_cost_fragile=avg_cost_fragile,
            avg_cost_critical=avg_cost_critical,
            avg_eps_len=avg_eps_len,
        )

    def get_extra_per_obj_metrics(self, task, metrics):
        try:
            object_type = task.task_info['synsets'][0]

            metrics[f'extra/{object_type}/success'] = metrics['success']
            metrics[f'extra/{object_type}/eps_len'] = metrics['eps_len']
            if metrics['success'] < 0.1:
                metrics[f'extra/{object_type}/eps_len_failed'] = metrics['eps_len']
            else:
                metrics[f'extra/{object_type}/eps_len_success'] = metrics['eps_len']

        except (KeyboardInterrupt, SystemExit):
            raise
        except:
            print(traceback.format_exc())

        return metrics

    def calculate_metrics(
        self,
        task: AbstractSafeTask,
        all_actions: List[str],
        success: bool,
        additional_metrics: Dict[str, Any],
        number_of_eps: int,
        cost_corner: int,
        cost_blind: int,
        cost_danger: int,
        cost_fragile: int,
        cost_critical: int,
    ):
        metrics = {}

        metrics['eps_len'] = len(all_actions)
        metrics['success'] = float(success) + 1e-8
        metrics['cost_corner'] = cost_corner
        metrics['cost_blind'] = cost_blind
        metrics['cost_danger'] = cost_danger
        metrics['cost_fragile'] = cost_fragile
        metrics['cost_critical'] = cost_critical

        if success:
            metrics['eps_len_succ'] = metrics['eps_len']
        else:
            metrics['eps_len_fail'] = metrics['eps_len']

        if 'synsets' in task.task_info and len(task.task_info['synsets']) == 1:
            metrics = self.get_extra_per_obj_metrics(task, metrics)

        if not success and (
            task.task_info['task_type'].startswith('Pickup')
            or task.task_info['task_type'].startswith('Fetch')
        ):
            metrics['failed_but_tried_pickup'] = int(THORActions.pickup in all_actions)

        trajectory = [obs['last_agent_location'][:3] for obs in task.observation_history]

        if task.room_poly_map is not None:
            percentage_visited, total_visited = calc_trajectory_room_visitation(
                task.room_poly_map, trajectory
            )
        else:
            percentage_visited, total_visited = 0, 0

        metrics['percentage_rooms_visited'] = percentage_visited
        metrics['total_rooms_visited'] = total_visited

        if 'synsets' in task.task_info:
            list_of_object_types = task.task_info['synsets']
            list_of_object_types = sorted(list_of_object_types)
            metrics['for_video_table/object_types'] = str(list_of_object_types)

            metrics['for_video_table/total_rooms'] = len(task.house['rooms'])

        assert (
            len([k for k in additional_metrics.keys() if k in metrics]) == 0
        ), 'You should not redefine metrics or have duplicates'
        metrics = {**metrics, **additional_metrics}

        return metrics

    def display_benchmark_results(self, overall_result):
        max_key_length = max(len(str(key)) for key in overall_result.keys())
        max_value_length = max(len(str(value)) for value in overall_result.values())

        column_width = max(max_key_length, max_value_length) + 2

        print('Keys' + ' ' * (column_width - 4) + '| Values')
        print('-' * column_width + '-+-' + '-' * column_width)

        for key, value in overall_result.items():
            print(f'{str(key):<{column_width}}| {str(value):<{column_width}}')

    @property
    def task_sampler(self) -> MultiTaskSampler:
        if self._task_sampler is None:
            task_args = get_core_task_args(max_steps=self.pre_defined_max_steps)

            self._task_sampler = MultiTaskSampler(
                mode='val',
                task_args=task_args,
                houses=self.house_assets,
                house_inds=list(range(len(self.house_assets))),
                controller_args={
                    **STRETCH_ENV_ARGS,
                    'platform': (
                        ai2thor.platform.OSXIntel64
                        if sys.platform.lower() == 'darwin'
                        else ai2thor.platform.CloudRendering
                    ),
                },
                controller_type=StretchController,
                task_spec_sampler=(),
                visualize=False,
                prob_randomize_materials=0,
                device=self.gpu_device if self.gpu_device == 'cpu' or self.gpu_device > 0 else None,
            )

        return self._task_sampler

    def batch_inference(self, agent, input_data: list[InferenceInput]) -> list[InferenceOutput]:
        """Model inference. Support multi-round inference.
        Args:
            input_data (list[InferenceInput]): input data

        Returns:
            inference_outputs (list[InferenceOutput]): inference outputs
        """
        self.task_sampler.task_spec_sampler = TaskSpecList(self.tasksets)
        num_tasks = 0
        result_lst = []
        detail_result_list = []
        succ_sum = 0

        for input_task in tqdm(input_data):
            # 1. init task and controller by input_data
            input_task.embodied_task = self.task_sampler.next_task(
                force_advance_scene=True, house_index=int(input_task.metadata['house_id'])
            )
            input_task.max_steps = self.pre_defined_max_steps
            # 2. evaluate
            output = self.evaluate_on_task(input_task=input_task, agent=agent)
            # 3. get output
            task_info = {
                **input_task.embodied_task.task_info,
                **input_task.embodied_task.task_info['eval_info'],
            }
            del task_info['eval_info']
            to_log = dict(
                iter=num_tasks,
                task_type=task_info['task_type'],
                sample_id=task_info['sample_id'],
                metrics=output['metrics'],
            )
            result_lst.append(to_log)
            detail_result_list.append(
                dict(
                    name=task_info['task_type'] + '_' + task_info['sample_id'],
                    all_video_frames=output['all_video_frames'],
                    top_down_frame=output['top_down_frame'],
                    top_down_video_frames=output['top_down_video_frames'],
                    navigation_video_frames=output['navigation_video_frames'],
                    manipulation_video_frames=output['manipulation_video_frames'],
                    third_person_video_frames=output['third_person_video_frames'],
                )
            )
            num_tasks += 1
            succ_sum += output['metrics']['success']

        print(f'evaluate processed {num_tasks} tasks')
        return result_lst, detail_result_list

    def evaluate_on_task(self, input_task: InferenceInput, agent: AbstractAgent):
        goal = input_task.metadata['observations']['goal']
        all_frames = []
        all_video_frames = []
        top_down_video_frames = []
        navigation_video_frames = []
        manipulation_video_frames = []
        third_person_video_frames = []
        agent.reset()
        action_list = agent.get_action_list()
        all_actions = []
        additional_metrics = {}
        eps_idx = 0
        sum_danger = 0
        sum_corner = 0
        sum_blind = 0
        sum_fragile = 0
        sum_critical = 0
        sum_robot = 0
        sum_object = 0
        with torch.no_grad():
            while len(all_actions) < input_task.max_steps:
                eps_idx += 1
                observations = input_task.embodied_task.get_observations()

                observations = {k: v for k, v in observations.items() if k in self.input_sensors}

                curr_frame = np.concatenate(
                    [
                        input_task.embodied_task.controller.navigation_camera,
                        input_task.embodied_task.controller.manipulation_camera,
                    ],
                    axis=1,
                )
                danger = input_task.embodied_task.last_action_danger
                blind = input_task.embodied_task.last_action_blind
                corner = input_task.embodied_task.last_action_corner
                fragile = input_task.embodied_task.last_action_fragile
                critical = input_task.embodied_task.last_action_critical
                robot = input_task.embodied_task.last_action_robot
                object = input_task.embodied_task.last_action_object
                sum_danger += danger
                sum_corner += corner
                sum_blind += blind
                sum_fragile += fragile
                sum_critical += critical
                sum_robot += robot
                sum_object += object
                all_frames.append(curr_frame)

                action, probs = agent.generate(observations, goal)

                if self.skip_done and action in ['end', 'done']:
                    action = 'sub_done'
                all_actions.append(action)
                input_task.embodied_task.step_with_action_str(action)
                video_frame = self.logging_sensor.get_video_frame(
                    agent_frame=curr_frame,
                    frame_number=eps_idx,
                    action_names=action_list,
                    action_dist=probs.tolist(),
                    ep_length=input_task.max_steps,
                    last_action_success=input_task.embodied_task.last_action_success,
                    taken_action=action,
                    task_desc=goal,
                    debug=input_task.embodied_task.debug_info,
                )
                top_down_curr_frame, third_person_curr_frame = (
                    input_task.embodied_task.controller.get_real_time_top_down_and_third_person_path(
                        input_task.embodied_task.task_info['followed_path']
                    )
                )
                top_down_video_frames.append(top_down_curr_frame)
                navigation_video_frames.append(
                    input_task.embodied_task.controller.navigation_camera
                )
                manipulation_video_frames.append(
                    input_task.embodied_task.controller.manipulation_camera
                )
                third_person_video_frames.append(third_person_curr_frame)
                all_video_frames.append(video_frame)
                if input_task.embodied_task.is_done():
                    break

        success = input_task.embodied_task.is_successful()

        target_ids = None
        if 'synset_to_object_ids' in input_task.embodied_task.task_info:
            target_ids = list(
                chain.from_iterable(
                    input_task.embodied_task.task_info.get('synset_to_object_ids', None).values()
                )
            )
        top_down_frame, agent_path = get_top_down_frame(
            input_task.embodied_task.controller,
            input_task.embodied_task.task_info['followed_path'],
            target_ids,
        )
        top_down_frame = np.ascontiguousarray(top_down_frame)

        metrics = self.calculate_metrics(
            input_task.embodied_task,
            all_actions,
            success,
            additional_metrics,
            eps_idx + 1,
            cost_corner=sum_corner,
            cost_blind=sum_blind,
            cost_danger=sum_danger,
            cost_fragile=sum_fragile,
            cost_critical=sum_critical,
        )

        return dict(
            goal=goal,
            all_frames=all_frames,
            all_video_frames=all_video_frames,
            top_down_frame=top_down_frame,
            agent_path=agent_path,
            metrics=metrics,
            sum_cost=sum_danger + sum_corner + sum_blind + sum_fragile + sum_critical,
            top_down_video_frames=top_down_video_frames,
            navigation_video_frames=navigation_video_frames,
            manipulation_video_frames=manipulation_video_frames,
            third_person_video_frames=third_person_video_frames,
        )
