# Copyright 2024 Allen Institute for AI
# ==============================================================================

import copy
import json
import math
from typing import Dict, List, Set, Tuple

import numpy as np
from scipy.spatial.transform import Rotation as R


def wrap_angle_to_pm180(angle):
    return (angle + 180) % 360 - 180


def angle_point_to_point(loc_start, loc_goal):
    # often loc_start = curr_state.base_position
    vector = (loc_goal['x'] - loc_start['x'], loc_goal['z'] - loc_start['z'])
    angle = math.degrees(math.atan2(vector[0], vector[1]))
    return wrap_angle_to_pm180(angle)


class StretchState:
    # Fixed values for stretch even for none instances
    arm_extreme_values = {
        'lift_max': 1.0457,
        'lift_min': -0.055,
        'lift_soft_min': 0.0,  # TODO
        'extend_max': 0.759,
        'extend_min': 0.243,
    }
    hand_length = 0.20  # projection into XZ plane for simplicity
    hand_height = 0.07  # approx 7cm between hand sphere center and bottom of wrist joint
    wrist_rotation_bounds = (75, 100)  # degrees
    agent_center_y_height = (
        0.9009982347488403  # prefer this to be a constant. Not fixated on it though.
    )

    arm_coord_offsets = {
        'translation': {'x': -0.1157, 'y': -0.739992, 'z': -0.0239},
        'rotation': {'theta': 90.0},  # AAGH
    }  # For transform between agent pivot and arm coordinate systems (measurable as specified in wrist_pose)

    max_interactable_height = (
        1.2  # 120cm - measured (approx) in real. Must be additional checks around viz.
    )

    def __init__(self, controller=None):
        if controller is not None:
            # if the controller has the property controller, use the sub_controller
            if hasattr(controller, 'controller'):
                controller = controller.controller
            wrist_pose = self.get_wrist_pose(controller)
            base_position = self.get_base_position(controller)
            absolute_hand_position = self.get_absolute_hand_position(controller)
            gripper_state = 0.0  # TODO this will be updated when THOR adds it to the metadata
            held_oids = {
                (True, oid) for oid in (controller.last_event.metadata['arm']['heldObjects'] or [])
            }

            # Initialize attributes
            self._base_position = {
                'x': base_position['x'],
                'y': self.agent_center_y_height,
                'z': base_position['z'],
                'theta': base_position['theta'],
            }
            self._wrist_pose = {
                'y': wrist_pose['y'],
                'z': wrist_pose['z'],
                'yaw': wrist_pose['yaw'],
            }
            # For sanity reasons, in wrist pose just don't bother translating the hand position to the base frame.
            # Arm Y is "lift" and arm Z is "extend". Use hand sphere for world reference and refer to
            # convert_world_to_arm_coordinate for the translation of world objects into wrist base coordinates.

            self._hand_position = {
                'x': absolute_hand_position['x'],
                'y': absolute_hand_position['y'],
                'z': absolute_hand_position['z'],
            }
            self._gripper_openness = gripper_state
            self._held_oids = held_oids
        else:
            # Default/non-privileged values for evaluation agent
            self._base_position = {
                'x': 0,
                'y': self.agent_center_y_height,
                'z': 0,
                'theta': 0,
            }
            self._wrist_pose = {'y': 0, 'z': 0, 'yaw': 0}
            self._hand_position = {'x': None, 'y': None, 'z': 0}
            self._gripper_openness = 0
            self._held_oids = set()

    @property
    def base_position(self) -> dict:
        return self._base_position

    @property
    def wrist_pose(self) -> dict:
        return self._wrist_pose

    @property
    def hand_position(self) -> dict:
        return self._hand_position

    @property
    def gripper_openness(self) -> float:
        # TODO: can't find this in THOR metadata. Have made feature request
        return 0

    @property
    def held_oids(self) -> Set[Tuple[bool, str]]:
        return self._held_oids

    def to_dict(self) -> Dict[str, Dict[str, float]]:
        return {
            'base_position': self.base_position,
            'wrist_pose': self.wrist_pose,
            'gripper_openness': self.gripper_openness,
            'held_oids': list(self.held_oids),
        }

    def __str__(self) -> str:
        return json.dumps(self.to_dict())

    @classmethod
    def signed_travel_distance_wrist(cls, initial_angle: float, final_angle: float):
        theta_bound_1, theta_bound_2 = cls.wrist_rotation_bounds

        # first, make everything 0 to 360
        initial_angle = initial_angle % 360
        final_angle = final_angle % 360

        # Check if the final angle is within the forbidden zone
        if theta_bound_1 <= final_angle <= theta_bound_2:
            # print(
            #     f"Warning: desired angle {final_angle} is in the forbidden zone. Setting to nearest boundary."
            # )
            if abs(final_angle - theta_bound_1) < abs(final_angle - theta_bound_2):
                final_angle = theta_bound_1
            else:
                final_angle = theta_bound_2

        # Calculate the signed travel distance considering the forbidden zone
        if final_angle > initial_angle:
            if initial_angle < theta_bound_1 and final_angle > theta_bound_2:
                signed_distance = final_angle - initial_angle - 360
            else:
                signed_distance = final_angle - initial_angle
        else:
            if initial_angle > theta_bound_2 and final_angle < theta_bound_1:
                signed_distance = final_angle - initial_angle + 360
            else:
                signed_distance = final_angle - initial_angle

        return signed_distance

    @classmethod
    def difference(cls, final_state, initial_state):
        diff_base = {}
        final_base_pos_in_initial_agent_frame = convert_world_to_agent_coordinate(
            final_state.base_position, initial_state, arm=False
        )

        for key in ['x', 'z', 'theta']:
            if final_state.base_position[key] is None or initial_state.base_position[key] is None:
                diff_base[key] = 0
            else:
                if key == 'theta':
                    diff_theta = final_state.base_position[key] - initial_state.base_position[key]
                    diff_base[key] = wrap_angle_to_pm180(diff_theta)
                else:
                    diff_base[key] = final_base_pos_in_initial_agent_frame[key]

        diff_wrist = {}
        for key in ['y', 'z', 'yaw']:
            if final_state.wrist_pose[key] is None or initial_state.wrist_pose[key] is None:
                diff_wrist[key] = 0
            else:
                diff_wrist[key] = final_state.wrist_pose[key] - initial_state.wrist_pose[key]
        if 'yaw' in diff_wrist:
            if (
                final_state.wrist_pose['yaw'] is not None
                and initial_state.wrist_pose['yaw'] is not None
            ):
                diff_wrist['yaw'] = cls.signed_travel_distance_wrist(
                    initial_state.wrist_pose['yaw'], final_state.wrist_pose['yaw']
                )
            else:
                diff_wrist['yaw'] = 0

        diff_hand = {}
        for key in final_state.hand_position.keys():
            if final_state.hand_position[key] is None or initial_state.hand_position[key] is None:
                diff_hand[key] = 0
            else:
                diff_hand[key] = final_state.hand_position[key] - initial_state.hand_position[key]

        diff_gripper = (
            0
            if final_state.gripper_openness is None or initial_state.gripper_openness is None
            else final_state.gripper_openness - initial_state.gripper_openness
        )

        diff_held_oids = cls._delta_held_oids(final_state, initial_state)

        # Create a new StretchState representing the difference
        return cls._create_difference_state(
            diff_base, diff_wrist, diff_hand, diff_gripper, diff_held_oids
        )

    @classmethod
    def _delta_held_oids(cls, after_state, before_state):
        # should return all the additions with True and all the deletions with False
        additions = after_state.held_oids - before_state.held_oids
        deletions = before_state.held_oids - after_state.held_oids
        return {(False, oid) for _, oid in deletions} | additions

    @classmethod
    def create_delta_from_goal(cls, current_state: 'StretchState', goal_dict):
        # use goal parameters and current state to create a goal state. goal_dict must have same keys as the state
        goal_state = copy.deepcopy(current_state)  # init from current
        if 'base_position' in goal_dict:
            # only do this for keys that match
            common_keys = set(goal_dict['base_position'].keys()).intersection(
                current_state.base_position.keys()
            )
            for key in common_keys:
                goal_state._base_position[key] = goal_dict['base_position'][key]
        if 'wrist_pose' in goal_dict:
            common_keys = set(goal_dict['wrist_pose'].keys()).intersection(
                current_state.wrist_pose.keys()
            )
            for key in common_keys:
                goal_state._wrist_pose[key] = goal_dict['wrist_pose'][key]
        if 'hand_position' in goal_dict:
            raise NotImplementedError(
                'Direct absolute hand position is not yet supported as a goal'
            )
        if 'gripper_openness' in goal_dict:
            goal_state._gripper_openness = goal_dict['gripper_openness']
        if 'held_oids' in goal_dict:
            goal_state._held_oids = goal_dict['held_oids']

        # if none of these keys are in the goal dict, what are you doing?
        if not any(
            [
                x in goal_dict
                for x in [
                    'base_position',
                    'wrist_pose',
                    'hand_position',
                    'gripper_openness',
                    'held_oids',
                ]
            ]
        ):
            raise ValueError(
                'goal_dict must have at least one of the following keys: base_position, wrist_pose, '
                'hand_position, gripper_openness, held_oids'
            )

        return cls.difference(final_state=goal_state, initial_state=current_state), goal_state

    @staticmethod
    def _create_difference_state(diff_base, diff_wrist, diff_hand, diff_gripper, diff_held_oids):
        # Helper method to create a new StretchState representing the difference
        difference_state = StretchState(None)  # Initialize with a dummy controller for now

        difference_state._base_position = diff_base
        difference_state._wrist_pose = diff_wrist
        difference_state._hand_position = diff_hand
        difference_state._gripper_openness = diff_gripper
        difference_state._held_oids = diff_held_oids

        return difference_state

    @classmethod
    def state_change_within_tolerance(
        cls, delta_state: 'StretchState', tolerance: 'StretchState'
    ) -> Tuple[bool, Dict[str, List[str]]]:
        """
        Check if the change between two StretchState instances is within the specified tolerance.
        The usecase is twofold:
        1. Check if an action space can actually act on the current desired state difference (e.g., if the arm is
        less than 1 cm away from the target, then the arm cannot get closer in the SPOC V1 action space).
        2. Check if the state change after an action has occurred counts as successful (agent moved enough).
        These two usecases will have different tolerances, one fixed and one a property of the action space.

        Parameters:
        - difference (StretchState): The difference between the two states.
        - tolerance (StretchState): Tolerance level for each attribute.

        Returns:
        - Tuple[bool, Dict[str, List[str]]]: True if the change is within tolerance, False otherwise.
          The dictionary contains lists of parameters that exceed the tolerances.
        """

        exceeding_params = {
            'base_position': [],
            'wrist_pose': [],
            'hand_position': [],
            'gripper_openness': [],
            'held_oids': [],
        }
        base_position_within_tolerance = True
        rss = math.sqrt(delta_state.base_position['x'] ** 2 + delta_state.base_position['z'] ** 2)
        threshold = math.sqrt(tolerance.base_position['x'] ** 2 + tolerance.base_position['z'] ** 2)
        if rss > threshold:
            exceeding_params['base_position'].extend(['x', 'z'])
            base_position_within_tolerance = False

        if abs(delta_state.base_position['theta']) > tolerance.base_position['theta']:
            exceeding_params['base_position'].append('theta')
            base_position_within_tolerance = False

        wrist_pose_within_tolerance = True
        for key in delta_state.wrist_pose.keys():
            if abs(delta_state.wrist_pose[key]) > tolerance.wrist_pose[key]:
                exceeding_params['wrist_pose'].append(key)
                wrist_pose_within_tolerance = False

        hand_position_within_tolerance = True
        for key in delta_state.hand_position.keys():
            if abs(delta_state.hand_position[key]) > tolerance.hand_position[key]:
                exceeding_params['hand_position'].append(key)
                hand_position_within_tolerance = False

        gripper_openness_within_tolerance = abs(
            delta_state.gripper_openness
        ) <= tolerance.gripper_openness or exceeding_params['gripper_openness'].append(
            'gripper_openness'
        )

        held_oids_within_tolerance = True
        if len(delta_state.held_oids) > 0:
            exceeding_params['held_oids'].extend(list(delta_state.held_oids))
            held_oids_within_tolerance = False

        return (
            base_position_within_tolerance
            and wrist_pose_within_tolerance
            and hand_position_within_tolerance
            and gripper_openness_within_tolerance
            and held_oids_within_tolerance,
            exceeding_params,
        )

    @staticmethod
    def get_base_position(controller):
        full_pos = controller.last_event.metadata['agent']

        return {
            'x': full_pos['position']['x'],
            'z': full_pos['position']['z'],
            'theta': full_pos['rotation']['y'],
        }

    @staticmethod
    def get_wrist_pose(controller):
        # pose because agent-relative
        # All of these offsets were hashed out with Winson to be specific and measurable against the real stretch
        # I am aware that this function is nuts
        # look it was a long punt to make the thor coordinate systems verifiable
        # TODO: add the beautiful reference drawings somewhere

        final_joint = controller.last_event.metadata['arm']['joints'][-1]
        assert final_joint['name'] == 'stretch_robot_wrist_2_jnt'
        vertical_distance_between_bottom_of_wrist_and_clamshell = (
            final_joint['rootRelativePosition']['y']
            - 0.07367  # remove distance inside robot body, between coord center and clamshell top
            - 0.0243  # remove extra height between wrist joint center and bottom of joint
        )
        # Maximum value per simulation: 'y': 1.0457
        # Minimum value per simulation (only with the arm extended): 'y': -0.055

        horizontal_distance_between_spine_flat_and_wrist_back = (
            final_joint['rootRelativePosition']['z']
            + 0.25946  # add the extra distance across the width of the robot body
            - 0.0163  # remove distance inside the robot body, between coord center and back of wrist
        )
        # Maximum value extension per simulation: 'z': 0.759
        # Minimum value extension per simulation: 'z': 0.243
        wrist_yaw = math.fmod(
            final_joint['rootRelativeRotation']['w'] * final_joint['rootRelativeRotation']['y'], 360
        )  # [-180,180]

        return {
            'y': vertical_distance_between_bottom_of_wrist_and_clamshell,
            'z': horizontal_distance_between_spine_flat_and_wrist_back,
            'yaw': wrist_yaw,
        }

    @staticmethod
    def get_absolute_hand_position(controller):
        return controller.last_event.metadata['arm']['handSphereCenter']

    def root_relative_hand_position(self, controller) -> dict:
        # TODO: this is getting added to the metadata return. Revisit when available.
        raise NotImplementedError


## Matrix and transform helper functions ##
def inverse_rot_trans_mat(mat):
    mat = np.linalg.inv(mat)
    return mat


def calc_inverse(deg):
    rotation = R.from_euler('xyz', [0, deg, 0], degrees=True)
    result = rotation.as_matrix()
    inverse = inverse_rot_trans_mat(result)
    return inverse


def make_rotation_matrix(position, rotation):
    result = np.zeros((4, 4))
    if rotation is None:
        rotation = dict(x=0, y=0, z=0)
    r = R.from_euler('xyz', [rotation['x'], rotation['y'], rotation['z']], degrees=True)
    result[:3, :3] = r.as_matrix()
    result[3, 3] = 1
    result[:3, 3] = [position['x'], position['y'], position['z']]
    return result


def position_rotation_from_mat(matrix):
    result = {'position': None, 'rotation': None}
    rotation = R.from_matrix(matrix[:3, :3]).as_euler('xyz', degrees=True)
    rotation_dict = {'x': rotation[0], 'y': rotation[1], 'z': rotation[2]}
    result['rotation'] = rotation_dict
    position = matrix[:3, 3]
    result['position'] = {'x': position[0], 'y': position[1], 'z': position[2]}
    return result


def convert_world_to_relative_coordinate(world_obj, relative_location, relative_rotation):
    agent_translation = [
        relative_location['x'],
        relative_location['y'] or 0,
        relative_location['z'],
    ]
    inverse_agent_rotation = calc_inverse(
        relative_rotation['theta']
    )  # This can be made faster by caching the inverse rotation matrices (there will be at most 360)
    if 'position' in world_obj and 'rotation' in world_obj:
        obj_matrix = make_rotation_matrix(world_obj['position'], world_obj['rotation'])
    else:
        obj_matrix = make_rotation_matrix(world_obj, rotation=None)
    obj_translation = np.matmul(inverse_agent_rotation, (obj_matrix[:3, 3] - agent_translation))
    # add rotation later
    obj_matrix[:3, 3] = obj_translation
    result = position_rotation_from_mat(obj_matrix)['position']
    return result


def convert_world_to_agent_coordinate(world_point, agent_state: StretchState, arm=False):
    agent_base_pos = {
        'x': agent_state.base_position['x'],
        'y': agent_state.agent_center_y_height,
        'z': agent_state.base_position['z'],
    }
    obj_in_agent = convert_world_to_relative_coordinate(
        world_point, agent_base_pos, agent_state.base_position
    )
    if arm:
        return convert_world_to_relative_coordinate(
            obj_in_agent,
            agent_state.arm_coord_offsets['translation'],
            agent_state.arm_coord_offsets['rotation'],
        )
    else:
        return obj_in_agent


def convert_relative_to_world_coordinate(agent_relative_point, agent_state: StretchState):
    agent_translation = np.array(
        [
            agent_state.base_position['x'],
            agent_state.base_position['y'] or 0,
            agent_state.base_position['z'],
        ]
    )
    agent_rotation = R.from_euler('y', agent_state.base_position['theta'], degrees=True).as_matrix()
    agent_relative_point = np.array(
        [agent_relative_point['x'], agent_relative_point['y'], agent_relative_point['z']]
    )
    world_point = np.matmul(agent_rotation, agent_relative_point) + agent_translation
    return {'x': world_point[0], 'y': world_point[1], 'z': world_point[2]}
