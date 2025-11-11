# Copyright 2024 Allen Institute for AI
# ==============================================================================

from typing import List

from allenact.utils.misc_utils import prepare_locals_for_super

from eval_anything.third_party.SPOC.tasks.abstract_task import AbstractSafeTask


try:
    pass
except ImportError:
    pass
#
import numpy as np
from allenact.utils.system import get_logger


class RewardShaper:
    def __init__(self, task: AbstractSafeTask) -> None:
        self.task = task
        self.task_info = task.task_info
        self.reward_config = task.reward_config
        self.action_names = task.action_names
        self.controller = task.controller

        self._rewards: List[float] = []
        self._distance_to_goal: List[float] = []

        self.distance_type = None
        self.dist_to_target_func = None

        self.reachable_positions = self.controller.get_reachable_positions()  # TODO: here or later?
        self.reachable_locations = self.get_reachable_locations()

    def get_reachable_locations(self):
        reachable_locs = [[pos['x'], pos['z']] for pos in self.reachable_positions]
        return np.array(reachable_locs).round(1)

    def get_agent_loc(self):
        agent_position = self.controller.get_current_agent_position()
        return round(agent_position['x'], 1), round(agent_position['z'], 1)

    def shaping(self) -> float:
        raise NotImplementedError


class ObjectNavRewardShaper(RewardShaper):
    def __init__(
        self,
        task: AbstractSafeTask,
    ) -> None:
        super().__init__(**prepare_locals_for_super(locals()))
        self.distance_type = self.task.distance_type
        self.dist_to_target_func = self.task.dist_to_target_func
        last_distance = self.dist_to_target_func()
        self.closest_distance = last_distance

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


class FetchRewardShaper(RewardShaper):
    def __init__(
        self,
        task: AbstractSafeTask,
    ) -> None:
        super().__init__(**prepare_locals_for_super(locals()))
        self.distance_type = self.task.distance_type

        self.dist_to_target_from_arm_func = self.min_l2_distance_to_target_from_arm
        self.dist_to_target_colliders_from_arm_func = (
            self.min_l2_distance_to_target_colliders_from_arm
        )

        self.last_distance_from_arm = self.dist_to_target_from_arm_func()
        self.last_distance_from_arm_to_colliders = self.dist_to_target_colliders_from_arm_func()
        self.closest_distance_from_arm_to_colliders = self.last_distance_from_arm_to_colliders

        self._took_pickup_action = False
        self.got_reward_for_pickup = False
        self.got_reward_for_pickupable = False

    def is_object_pickupable(self):
        pickupable_object = self.controller.get_objects_in_hand_sphere()
        object_type = self.task_info['synsets'][0]
        for object_id in self.task_info['synset_to_object_ids'][object_type]:
            if object_id in pickupable_object:
                return True
        return False

    def min_l2_distance_to_target_from_arm(self) -> float:
        """Return the minimum distance to a target object.

        May return a negative value if the target object is not reachable.
        """
        # NOTE: may return -1 if the object is unreachable.
        min_dist = float('inf')
        object_type = self.task_info['synsets'][0]
        for object_id in self.task_info['synset_to_object_ids'][object_type]:
            min_dist = min(
                min_dist,
                self.controller.dist_from_arm_sphere_center_to_obj(object_id),
            )
        if min_dist == float('inf'):
            get_logger().error(
                f"No target object among {self.task_info['synset_to_object_ids'][object_type]} found"
                f" in house {self.task_info['house_index']}."
            )
            return -1.0
        return min_dist  #  This is only for the first part of the task but close enough

    def min_l2_distance_to_target_colliders_from_arm(self) -> float:
        """Return the minimum distance to a target object.

        May return a negative value if the target object is not reachable.
        """
        # NOTE: may return -1 if the object is unreachable.
        min_dist = float('inf')
        object_type = self.task_info['synsets'][0]
        for object_id in self.task_info['synset_to_object_ids'][object_type]:
            min_dist = min(
                min_dist,
                self.controller.dist_from_arm_sphere_center_to_obj_colliders_closest_to_point(
                    object_id
                ),
            )
        if min_dist == float('inf'):
            get_logger().error(
                f"No target object among {self.task_info['synset_to_object_ids'][object_type]} found"
                f" in house {self.task_info['house_index']}."
            )
            return -1.0
        return min_dist  #  This is only for the first part of the task but close enough

    def shaping(self) -> float:
        if self.reward_config is None:
            return 0
        if self.reward_config.shaping_weight == 0.0:
            return 0

        reward = 0.0

        if (
            not self.got_reward_for_pickup
            and self._took_pickup_action
            and self.task.successful_if_done()
        ):
            reward += 5.0
            self.got_reward_for_pickup = True

        if not self.got_reward_for_pickupable and self.is_object_pickupable():
            reward += 5.0
            self.got_reward_for_pickupable = True

        # distance reward
        cur_distance = self.dist_to_target_colliders_from_arm_func()
        if self.distance_type == 'l2':
            reward += (
                self.reward_config.shaping_weight
                * 5
                * max(self.closest_distance_from_arm_to_colliders - cur_distance, 0)
            )
            self.closest_distance_from_arm_to_colliders = min(
                self.closest_distance_from_arm_to_colliders, cur_distance
            )

        return reward


class RoomVisitRewardShaper(RewardShaper):
    def __init__(
        self,
        task: AbstractSafeTask,
    ) -> None:
        super().__init__(**prepare_locals_for_super(locals()))

    def shaping(self) -> float:
        if self.reward_config is None:
            return 0
        if self.reward_config.shaping_weight == 0.0:
            return 0

        reward = 0.0

        if len(self.task.seen_rooms) > self.task.last_num_seen_rooms:
            self.task.last_num_seen_rooms = len(self.task.seen_rooms)

        locs_index = (
            ((self.reachable_locations - np.array(self.get_agent_loc())) ** 2).sum(axis=1).argmin()
        )
        cur_loc = tuple(self.reachable_locations[locs_index])
        if cur_loc not in self.task.visited_loc:
            reward += 0.005
            self.task.visited_loc.add(cur_loc)

        if self.task.get_current_room() not in self.task.visited_rooms:
            reward += 2.0
            self.task.visited_rooms.add(self.task.get_current_room())

        if self.task._took_sub_done_action:
            if self.task.last_action_success:
                reward += 2.0
            else:
                reward -= 0.2

        return reward * self.reward_config.shaping_weight
