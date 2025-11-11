# Copyright 2024 Allen Institute for AI
# ==============================================================================

import os

import ai2thor.fifo_server

from eval_anything.third_party.SPOC.utils.constants.objaverse_data_dirs import OBJAVERSE_ASSETS_DIR
from eval_anything.third_party.SPOC.utils.type_utils import THORActions


STRETCH_COMMIT_ID = '966bd7758586e05d18f6181f459c0e90ba318bec'

try:
    from ai2thor.hooks.procedural_asset_hook import (
        ProceduralAssetHookRunner,
        create_assets_if_not_exist,
        get_all_asset_ids_recursively,
    )
except ImportError:
    raise ImportError(
        'Cannot import `ProceduralAssetHookRunner`. Please install the appropriate version of ai2thor:\n'
        f'```\npip install --extra-index-url https://ai2thor-pypi.allenai.org'
        f' ai2thor==0+{STRETCH_COMMIT_ID}\n```'
    )

AGENT_ROTATION_DEG = 30
AGENT_MOVEMENT_CONSTANT = 0.2
HORIZON = 0  # RH: Do not change from 0! this is now set elsewhere with RotateCameraMount actions
ARM_MOVE_CONSTANT = 0.1
WRIST_ROTATION = 10

EMPTY_BBOX = [1000, 1000, 1000, 1000, 0]
EMPTY_DOUBLE_BBOX = EMPTY_BBOX + EMPTY_BBOX

ORIGINAL_INTEL_W, ORIGINAL_INTEL_H = 1280, 720
INTEL_CAMERA_WIDTH, INTEL_CAMERA_HEIGHT = 396, 224


INTEL_WIDTH_CROPPED, INTEL_HEIGHT_CROPPED = 384, 224
INTEL_VERTICAL_FOV = 59
AGENT_RADIUS_LIST = [(0, 0.5), (1, 0.4), (2, 0.3), (3, 0.2)]

MAXIMUM_SERVER_TIMEOUT = 1200  # default : 100 Need to increase this for cloudrendering


class ProceduralAssetHookRunnerResetOnNewHouse(ProceduralAssetHookRunner):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.last_asset_id_set = set()

    def Initialize(self, action, controller):
        if self.asset_limit > 0:
            return controller.step(
                action='DeleteLRUFromProceduralCache', assetLimit=self.asset_limit
            )

    def CreateHouse(self, action, controller):
        house = action['house']
        asset_ids = get_all_asset_ids_recursively(house['objects'], [])
        asset_ids_set = set(asset_ids)
        if not asset_ids_set.issubset(self.last_asset_id_set):
            controller.step(action='DeleteLRUFromProceduralCache', assetLimit=0)
            self.last_asset_id_set = set(asset_ids)

        if STRETCH_COMMIT_ID == '5e43486351ac6339c399c199e601c9dd18daecc3':
            return create_assets_if_not_exist(
                controller=controller,
                asset_ids=asset_ids,
                asset_directory=self.asset_directory,
                asset_symlink=self.asset_symlink,
                stop_if_fail=self.stop_if_fail,
            )
        else:
            return create_assets_if_not_exist(
                controller=controller,
                asset_ids=asset_ids,
                asset_directory=self.asset_directory,
                asset_symlink=self.asset_symlink,
                stop_if_fail=self.stop_if_fail,
                copy_to_dir=os.path.join(controller._build.base_dir, self.target_dir),
                load_file_in_unity=False,
            )


_ACTION_HOOK_RUNNER = ProceduralAssetHookRunnerResetOnNewHouse(
    asset_directory=OBJAVERSE_ASSETS_DIR, asset_symlink=True, verbose=False, asset_limit=200
)

PHYSICS_SETTLING_TIME = 1.0

MAXIMUM_DISTANCE_ARM_FROM_AGENT_CENTER = (
    0.8673349051766235  # Computed with fixed arm agent, should have pairity with real
)

SAVE_DEPTH = False

STRETCH_ENV_ARGS = dict(
    gridSize=AGENT_MOVEMENT_CONSTANT
    * 0.75,  # Intentionally make this smaller than AGENT_MOVEMENT_CONSTANT to improve fidelity
    width=INTEL_CAMERA_WIDTH,
    height=INTEL_CAMERA_HEIGHT,
    # quality="Ultra",
    visibilityDistance=MAXIMUM_DISTANCE_ARM_FROM_AGENT_CENTER,
    visibilityScheme='Distance',
    fieldOfView=INTEL_VERTICAL_FOV,
    server_class=ai2thor.fifo_server.FifoServer,
    useMassThreshold=False,
    massThreshold=10,
    autoSimulation=False,
    autoSyncTransforms=True,
    renderInstanceSegmentation=True,
    agentMode='stretch',
    renderDepthImage=SAVE_DEPTH,
    cameraNearPlane=0.01,  # VERY VERY IMPORTANT
    branch=None,  # IMPORTANT do not use branch
    commit_id=STRETCH_COMMIT_ID,
    server_timeout=MAXIMUM_SERVER_TIMEOUT,
    snapToGrid=False,
    fastActionEmit=True,
    action_hook_runner=_ACTION_HOOK_RUNNER,
    render_mani_camera=True,
    use_quick_navi_action=True,
)

assert (
    STRETCH_ENV_ARGS.get('branch') is None and STRETCH_ENV_ARGS['commit_id'] is not None
), 'Should always specify the commit id and not the branch.'


ADDITIONAL_ARM_ARGS = {
    'returnToStart': True,
    'speed': 1,
}

ADDITIONAL_NAVIGATION_ARGS = {
    **ADDITIONAL_ARM_ARGS,
    'returnToStart': False,
}

STRETCH_WRIST_BOUND_1 = 75
STRETCH_WRIST_BOUND_2 = -260

if os.getenv('ACTION_DICT') is not None:
    import json

    assert os.path.exists(os.getenv('ACTION_DICT'))
    ALL_STRETCH_ACTIONS = list(json.load(open(os.getenv('ACTION_DICT'))).keys())
else:
    ALL_STRETCH_ACTIONS = [
        THORActions.move_ahead,
        THORActions.rotate_right,
        THORActions.rotate_left,
        THORActions.move_back,
        THORActions.done,
        THORActions.sub_done,
        THORActions.rotate_left_small,
        THORActions.rotate_right_small,
        THORActions.pickup,
        THORActions.move_arm_in,
        THORActions.move_arm_out,
        THORActions.move_arm_up,
        THORActions.move_arm_down,
        THORActions.wrist_open,
        THORActions.wrist_close,
        THORActions.move_arm_down_small,
        THORActions.move_arm_in_small,
        THORActions.move_arm_out_small,
        THORActions.move_arm_up_small,
        THORActions.dropoff,
    ]


##
# actions = [move ahead, move back, left , right, done,...]

stretch_long_names = {
    THORActions.move_ahead: 'move_ahead',
    THORActions.rotate_right: 'rotate_right',
    THORActions.rotate_left: 'rotate_left',
    THORActions.move_back: 'move_back',
    THORActions.done: 'done',
    THORActions.sub_done: 'sub_done',
    THORActions.rotate_left_small: 'rotate_left_small',
    THORActions.rotate_right_small: 'rotate_right_small',
    THORActions.pickup: 'pickup',
    THORActions.dropoff: 'dropoff',
    THORActions.move_arm_in: 'move_arm_in',
    THORActions.move_arm_out: 'move_arm_out',
    THORActions.move_arm_up: 'move_arm_up',
    THORActions.move_arm_down: 'move_arm_down',
    THORActions.wrist_open: 'wrist_open',
    THORActions.wrist_close: 'wrist_close',
    THORActions.move_arm_down_small: 'move_arm_down_small',
    THORActions.move_arm_in_small: 'move_arm_in_small',
    THORActions.move_arm_out_small: 'move_arm_out_small',
    THORActions.move_arm_up_small: 'move_arm_up_small',
}

if os.getenv('LONG_ACTION_NAME') is not None and bool(int(os.getenv('LONG_ACTION_NAME'))):
    ALL_STRETCH_ACTIONS = [stretch_long_names[short_name] for short_name in ALL_STRETCH_ACTIONS]

robot_action_mapping = {
    THORActions.move_ahead: {
        'action': 'MoveAgent',
        'args': {'move_scalar': AGENT_MOVEMENT_CONSTANT},
    },
    THORActions.move_back: {
        'action': 'MoveAgent',
        'args': {'move_scalar': -AGENT_MOVEMENT_CONSTANT},
    },
    THORActions.rotate_right: {
        'action': 'RotateAgent',
        'args': {'move_scalar': AGENT_ROTATION_DEG},
    },
    THORActions.rotate_left: {
        'action': 'RotateAgent',
        'args': {'move_scalar': -AGENT_ROTATION_DEG},
    },
    THORActions.rotate_right_small: {
        'action': 'RotateAgent',
        'args': {'move_scalar': AGENT_ROTATION_DEG / 5},
    },
    THORActions.rotate_left_small: {
        'action': 'RotateAgent',
        'args': {'move_scalar': -AGENT_ROTATION_DEG / 5},
    },
    THORActions.done: {'action': 'Pass', 'args': {}},
    THORActions.sub_done: {'action': 'Pass', 'args': {}},
    THORActions.move_arm_up: {'action': 'MoveArmBase', 'args': {'move_scalar': ARM_MOVE_CONSTANT}},
    THORActions.move_arm_up_small: {
        'action': 'MoveArmBase',
        'args': {'move_scalar': ARM_MOVE_CONSTANT / 5},
    },
    THORActions.move_arm_down: {
        'action': 'MoveArmBase',
        'args': {'move_scalar': -ARM_MOVE_CONSTANT},
    },
    THORActions.move_arm_down_small: {
        'action': 'MoveArmBase',
        'args': {'move_scalar': -ARM_MOVE_CONSTANT / 5},
    },
    THORActions.move_arm_out: {
        'action': 'MoveArmExtension',
        'args': {'move_scalar': ARM_MOVE_CONSTANT},
    },
    THORActions.move_arm_out_small: {
        'action': 'MoveArmExtension',
        'args': {'move_scalar': ARM_MOVE_CONSTANT / 5},
    },
    THORActions.move_arm_in: {
        'action': 'MoveArmExtension',
        'args': {'move_scalar': -ARM_MOVE_CONSTANT},
    },
    THORActions.move_arm_in_small: {
        'action': 'MoveArmExtension',
        'args': {'move_scalar': -ARM_MOVE_CONSTANT / 5},
    },
    THORActions.wrist_open: {'action': 'MoveWrist', 'args': {'move_scalar': -WRIST_ROTATION}},
    THORActions.wrist_close: {'action': 'MoveWrist', 'args': {'move_scalar': WRIST_ROTATION}},
    THORActions.pickup: {'action': 'GraspTo', 'args': {'move_to': -10}},
    THORActions.dropoff: {'action': 'GraspTo', 'args': {'move_to': 30}},
}
