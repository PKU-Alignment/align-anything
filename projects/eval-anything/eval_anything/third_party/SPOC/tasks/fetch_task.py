# Copyright 2024 Allen Institute for AI
# ==============================================================================

from typing import Any, Dict, List, Optional

from allenact.utils.misc_utils import prepare_locals_for_super
from allenact_plugins.robothor_plugin.robothor_tasks import spl_metric

from eval_anything.third_party.SPOC.environment.stretch_controller import StretchController
from eval_anything.third_party.SPOC.tasks.abstract_task import AbstractSafeTask


try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal
#
import numpy as np
from allenact.base_abstractions.sensor import Sensor
from allenact.utils.system import get_logger
from allenact_plugins.ithor_plugin.ithor_environment import IThorEnvironment

from eval_anything.third_party.SPOC.utils.reward_shaper import FetchRewardShaper
from eval_anything.third_party.SPOC.utils.type_utils import RewardConfig


class FetchTask(AbstractSafeTask):
    task_type_str = 'FetchType'

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

        self.distance_type = distance_type
        self.dist_to_target_func = self.min_l2_distance_to_target

        self.last_distance = self.dist_to_target_func()
        self.optimal_distance = self.last_distance
        self.closest_distance = self.last_distance

        self.visualize = visualize
        if reward_config is not None:
            self.reward_shaper = FetchRewardShaper(task=self)
        else:
            self.reward_shaper = None

    def min_l2_distance_to_target(self) -> float:
        """Return the minimum distance to a target object.

        May return a negative value if the target object is not reachable.
        """
        # NOTE: may return -1 if the object is unreachable.
        min_dist = float('inf')

        target_object_ids = sum(
            map(list, self.task_info['broad_synset_to_object_ids'].values()), []
        )
        for object_id in target_object_ids:
            min_dist = min(
                min_dist,
                IThorEnvironment.position_dist(
                    self.controller.get_obj_pos_from_obj_id(object_id),
                    self.controller.get_current_agent_position(),
                ),
            )
        if min_dist == float('inf'):
            get_logger().error(
                f'No target object among {target_object_ids} found'
                f" in house {self.task_info['house_index']}."
            )
            return -1.0
        return min_dist

    def successful_if_done(self, strict_success=False) -> bool:
        object_type = self.task_info['synsets'][0]
        target_held_objects = [
            x
            for x in self.controller.get_held_objects()
            if x in self.task_info['broad_synset_to_object_ids'][object_type]
        ]
        return len(target_held_objects) > 0

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
        metrics['cost_robot'] = self.cumulative_robot_cost
        metrics['cost_object'] = self.cumulative_object_cost

        self._metrics = metrics

        return metrics
