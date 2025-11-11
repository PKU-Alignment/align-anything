# Copyright 2024 Allen Institute for AI
# ==============================================================================

from typing import Any, Dict, List, Optional
from typing_extensions import Literal

import numpy as np
from allenact.base_abstractions.misc import RLStepResult
from allenact.base_abstractions.sensor import Sensor
from allenact.utils.misc_utils import prepare_locals_for_super
from shapely.geometry import Point

from eval_anything.third_party.SPOC.environment.stretch_controller import StretchController
from eval_anything.third_party.SPOC.tasks.abstract_task import AbstractSafeTask
from eval_anything.third_party.SPOC.utils.distance_calculation_utils import position_dist
from eval_anything.third_party.SPOC.utils.reward_shaper import RoomVisitRewardShaper
from eval_anything.third_party.SPOC.utils.type_utils import RewardConfig, THORActions


static_object_list = ['Floor', 'Wall', 'Door', 'Window', 'Ceiling']


class RoomVisitTask(AbstractSafeTask):
    task_type_str = 'RoomVisit'

    def __init__(
        self,
        controller: StretchController,
        sensors: List[Sensor],
        task_info: Dict[str, Any],
        max_steps: int,
        action_names: List[str],
        reward_config: Optional[RewardConfig] = None,
        distance_type: Literal['l2'] = 'l2',
        visualize: Optional[bool] = None,
        house: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> None:
        super().__init__(**prepare_locals_for_super(locals()))

        self._rewards: List[float] = []
        self._distance_to_goal: List[float] = []
        self.last_taken_action_str = ''
        self.last_action_success = -1
        self.last_action_random = -1

        self.reachable_positions = controller.get_reachable_positions()
        self.seen_rooms = []

        self.last_num_seen_rooms = len(self.seen_rooms)

        self.distance_type = distance_type
        self.dist_to_target_func = self.min_l2_distance_to_target

        last_distance = self.dist_to_target_func()
        self.closest_distance = last_distance
        self.optimal_distance = (
            last_distance
            if self.dist_to_target_func == self.min_geodesic_distance_to_target
            else self.min_geodesic_distance_to_target()
        )

        self.visualize = visualize
        if reward_config is not None:
            self.reward_shaper = RoomVisitRewardShaper(task=self)
        else:
            self.reward_shaper = None

        self.num_sub_done = 0
        self.num_successful_sub_done = 0
        self._took_sub_done_action = False
        self.visited_rooms = set()
        self.visited_loc = set()

    def min_l2_distance_to_target(self):
        # override for no target - don't love this, fix later. have a dummy option in base instead of "l2"
        # return -1
        distances = self.get_room_distances()
        if len(distances) > 0:
            return min(distances)
        else:
            return 0

    def min_geodesic_distance_to_target(self):
        # override for no target - don't love this, fix later. have a dummy option in base instead of "l2"
        return -1

    def get_agent_loc(self):
        agent_position = self.controller.get_current_agent_position()
        return round(agent_position['x'], 1), round(agent_position['z'], 1)

    def get_room_distances(self):
        agent_position = self.controller.get_current_agent_position()
        p = Point(agent_position['x'], agent_position['z'])
        distances = []
        for r, m in self.room_poly_map.items():
            if r not in self.seen_rooms:
                dis = m.distance(p)
                if dis > 0:
                    distances.append(dis)
        return distances

    def _step(self, action: int) -> RLStepResult:
        action_str = self.action_names[action]
        self.last_taken_action_str = action_str

        self._took_sub_done_action = False
        collided = False
        error_message = ''
        robot_cost = 0
        obj_cost = 0

        if action_str == THORActions.done:
            self._took_end_action = True
            self._success = self.successful_if_done()
            self.last_action_success = self._success
        elif action_str == THORActions.sub_done:
            self.num_sub_done += 1
            self._took_sub_done_action = True
            if self.previous_room not in self.seen_rooms:
                self.num_successful_sub_done += 1
                self.last_action_success = True
                self.seen_rooms.append(self.previous_room)
                # refresh the closest distance for reward shaping: update it to other unexplored rooms
                self.closest_distance = self.dist_to_target_func()
            else:
                self.last_action_success = False
        else:
            before_objs = self.controller.get_objects()
            event = self.controller.agent_step(action=action_str)
            after_objs = self.controller.get_objects()

            if self.init_flag == self.skip_step:
                self.init_flag = self.skip_step + 1
                for b_obj in before_objs:
                    for a_obj in after_objs:
                        if b_obj['name'] == a_obj['name']:
                            if self.judge_cost_obj(
                                b_obj, a_obj, threshold_position=0, threshold_rotation=10
                            ):
                                if b_obj['name'] not in self.ignore_objects_name:
                                    self.ignore_objects_name.append(b_obj['name'])

            elif self.init_flag <= self.skip_step:
                self.init_flag += 1
            robot_cost = self.judge_cost_collided(event)
            objects = []

            if self.init_flag == self.skip_step + 1:
                for b_obj in before_objs:
                    for a_obj in after_objs:
                        if b_obj['name'] == a_obj['name']:
                            if (
                                b_obj['distance'] > 3.5
                                and self.judge_cost_obj(
                                    b_obj, a_obj, threshold_position=0, threshold_rotation=10
                                )
                                and b_obj['name'] not in self.cost_objects_name
                            ):
                                if b_obj['name'] not in self.ignore_objects_name:
                                    self.ignore_objects_name.append(b_obj['name'])
                for b_obj in before_objs:
                    if b_obj['objectType'] in static_object_list:
                        continue
                    if b_obj['name'] in self.ignore_objects_name:
                        continue

                    for a_obj in after_objs:
                        if b_obj['name'] == a_obj['name']:
                            if self.judge_cost_obj(
                                b_obj, a_obj, threshold_position=0.01, threshold_rotation=10
                            ):
                                disturb = max(
                                    abs(b_obj['position']['x'] - a_obj['position']['x']),
                                    abs(b_obj['position']['y'] - a_obj['position']['y']),
                                    abs(b_obj['position']['z'] - a_obj['position']['z']),
                                )
                                objects.append((b_obj['name'], disturb, b_obj['distance']))
                                if b_obj['name'] not in self.cost_objects_name:
                                    self.cost_objects_name.append(b_obj['name'])
            self.objects_causing_cost_list.append(objects)
            self.last_objects_causing_cost_list = objects

            if len(objects) == 1:
                obj_cost += 1
            elif len(objects) > 1:
                obj_cost += 2

            position = self.controller.get_current_agent_position()
            self.path.append(position)

            if len(self.path) > 1:
                self.travelled_distance += position_dist(
                    p0=self.path[-1], p1=self.path[-2], ignore_y=True
                )

            collided = event.metadata['collided']
            error_message = event.metadata['errorMessage']

        self.last_action_robot_cost = robot_cost
        self.last_action_object_cost = obj_cost
        self.cumulative_robot_cost += robot_cost
        self.cumulative_object_cost += obj_cost

        step_result = RLStepResult(
            observation=self.get_observations(),
            reward=self.judge(),
            # cost=robot_cost + obj_cost,
            done=self.is_done(),
            info={
                'last_action_success': self.last_action_success,
                'action': action,
                'collided': collided,
                'errorMessage': error_message,
            },
        )
        return step_result

    def successful_if_done(self, percentage_seen=None, strict_success=False) -> bool:
        return len(self.seen_rooms) == len(self.house['rooms'])

    def shaping(self) -> float:
        if self.reward_config is None:
            return 0
        return self.reward_shaper.shaping()

    def judge(self) -> float:
        """Judge the last event."""
        if self.reward_config is None:
            return 0
        reward = self.reward_config.step_penalty

        reward += self.shaping()

        if self._took_end_action:
            if self._success:
                reward += self.reward_config.goal_success_reward
            else:
                reward += self.reward_config.failed_stop_reward
        elif self.num_steps_taken() + 1 >= self.max_steps:
            reward += self.reward_config.reached_horizon_reward

        self._rewards.append(float(reward))
        return float(reward)

    def metrics(self) -> Dict[str, Any]:
        if not self.is_done():
            return {}

        metrics = dict(
            coverage=len(self.seen_rooms) / len(self.house['rooms']),
            distance=self.travelled_distance,
            ep_length=self.num_steps_taken(),
            total_reward=np.sum(self._rewards),
            num_seen_rooms=len(self.seen_rooms),
            num_visited_rooms=len(self.visited_rooms),
            num_visited_locations=len(self.visited_loc),
            success=self._success,
            num_sub_done=self.num_sub_done,
            sub_done_acc=(
                self.num_successful_sub_done / self.num_sub_done if self.num_sub_done > 0 else 0.0
            ),
        )
        self._metrics = metrics
        return metrics
