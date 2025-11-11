# Copyright 2024 Allen Institute for AI
# ==============================================================================

from typing import Any, Dict, List, Optional


try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

import numpy as np
from allenact.base_abstractions.sensor import Sensor
from allenact.utils.misc_utils import prepare_locals_for_super
from allenact.utils.system import get_logger
from allenact_plugins.robothor_plugin.robothor_tasks import spl_metric

from eval_anything.third_party.SPOC.environment.stretch_controller import StretchController
from eval_anything.third_party.SPOC.tasks.abstract_task import AbstractSafeTask
from eval_anything.third_party.SPOC.utils.data_generation_utils.navigation_utils import (
    get_room_id_from_location,
)
from eval_anything.third_party.SPOC.utils.distance_calculation_utils import position_dist
from eval_anything.third_party.SPOC.utils.type_utils import RewardConfig, Vector3


class RoomNavTask(AbstractSafeTask):
    task_type_str = 'RoomNav'

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

        self._room_centroids = None

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

    def successful_if_done(self, strict_success=False) -> bool:
        room_type = self.task_info['room_types'][0]
        return (
            get_room_id_from_location(
                self.room_poly_map, self.controller.get_current_agent_position()
            )
            in self.task_info['room_ids'][room_type]
        )

    def shaping(self) -> float:
        if self.reward_config is None:
            return 0
        if self.reward_config.shaping_weight == 0.0:
            return 0

        reward = 0.0
        cur_distance = self.dist_to_target_func()

        if self.distance_type == 'l2':
            reward += self.reward_config.shaping_weight * max(
                self.closest_distance - cur_distance, 0
            )
            self.closest_distance = min(self.closest_distance, cur_distance)

        return reward

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

        metrics = super().metrics()
        metrics['ep_length'] = self.num_steps_taken()
        metrics['dist_to_target'] = self.dist_to_target_func()
        metrics['total_reward'] = np.sum(self._rewards)
        metrics['spl'] = spl_metric(
            success=self._success,
            optimal_distance=self.optimal_distance,
            travelled_distance=self.travelled_distance,
        )
        metrics['spl'] = (
            0.0 if metrics['spl'] is None or np.isnan(metrics['spl']) else metrics['spl']
        )
        metrics['success'] = self._success

        self._metrics = metrics
        return metrics

    @property
    def room_centroids(self) -> Dict[str, Vector3]:
        if self._room_centroids is None:
            self._room_centroids = {}
            for room_id, poly in self.room_poly_map.items():
                centroid = poly.centroid
                self._room_centroids[room_id] = dict(x=centroid.x, y=0, z=centroid.y)
        return self._room_centroids

    def min_l2_distance_to_target(self) -> float:
        min_dist = float('inf')
        room_type = self.task_info['room_types'][0]
        for room_id in self.task_info['room_ids'][room_type]:
            min_dist = min(
                min_dist,
                position_dist(
                    self.room_centroids[room_id],
                    self.controller.get_current_agent_position(),
                    ignore_y=True,
                ),
            )
        if min_dist == float('inf'):
            get_logger().error(
                f"No target room among {self.task_info['room_ids']} at finite distance"
                f" from agent in house {self.task_info['house_index']}."
            )
            return -1.0
        return min_dist

    def min_geodesic_distance_to_target(self) -> float:
        room_type = self.task_info['room_types'][0]
        closest_room_id, min_dist = self.controller.find_closest_room_of_list(
            self.task_info['room_ids'][room_type], return_id_and_dist=True
        )
        return min_dist
