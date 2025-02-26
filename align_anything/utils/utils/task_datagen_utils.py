import sys
from typing import Optional, Sequence

from allenact.base_abstractions.sensor import Sensor

from environment.manipulation_sensors import AnObjectIsInHand, RelativeArmLocationMetadata
from environment.navigation_sensors import (
    LastActionSuccessSensor,
    LastActionIsRandomSensor,
    LastAgentLocationSensor,
    LastActionStrSensor,
    HouseNumberSensor,
    SlowAccurateObjectBBoxSensor,
    TaskTemplatedTextSpecSensor,
    HypotheticalTaskSuccessSensor,
    MinimumTargetAlignmentSensor,
    Visible4mTargetCountSensor,
    MinL2TargetDistanceSensor,
    RoomsSeenSensor,
    RoomCurrentSeenSensor,
    TaskRelevantObjectBBoxSensor,
)
from environment.vision_sensors import (
    RawNavigationStretchRGBSensor,
    RawManipulationStretchRGBSensor,
)
from utils.constants.stretch_initialization_utils import (
    INTEL_CAMERA_WIDTH,
    INTEL_CAMERA_HEIGHT,
    ALL_STRETCH_ACTIONS,
)
from utils.type_utils import AbstractTaskArgs

NUM_WORKERS_ON_SINGLE_DEVICE = (
    3 if sys.platform == "linux" else 1
)  # Talk to KE or LW if you are thinking of increasing this

_DATASET_CACHE = {}


def get_core_sensors():
    return [
        RawNavigationStretchRGBSensor(
            uuid="raw_navigation_camera",
            width=INTEL_CAMERA_WIDTH,
            height=INTEL_CAMERA_HEIGHT,
        ),
        RawManipulationStretchRGBSensor(
            uuid="raw_manipulation_camera",
            width=INTEL_CAMERA_WIDTH,
            height=INTEL_CAMERA_HEIGHT,
        ),
        # THIS IS VERY VERY SLOW
        # TopDownPathViewRGBSensor(
        #     uuid="path_view_camera",
        #     width=INTEL_CAMERA_WIDTH,
        #     height=INTEL_CAMERA_HEIGHT,
        # ),
        LastActionSuccessSensor(),
        LastActionIsRandomSensor(),
        LastAgentLocationSensor(),
        LastActionStrSensor(),
        HouseNumberSensor(),
        TaskTemplatedTextSpecSensor(),
        HypotheticalTaskSuccessSensor(),
        MinimumTargetAlignmentSensor(),
        Visible4mTargetCountSensor(),
        TaskRelevantObjectBBoxSensor(which_camera="nav", uuid="nav_task_relevant_object_bbox"),
        TaskRelevantObjectBBoxSensor(which_camera="manip", uuid="manip_task_relevant_object_bbox"),
        SlowAccurateObjectBBoxSensor(which_camera="nav", uuid="nav_accurate_object_bbox"),
        SlowAccurateObjectBBoxSensor(which_camera="manip", uuid="manip_accurate_object_bbox"),
        MinL2TargetDistanceSensor(),
        RoomCurrentSeenSensor(),
        RoomsSeenSensor(),
        AnObjectIsInHand(),
        RelativeArmLocationMetadata(),
    ]


def get_core_task_args(max_steps: int, core_sensors=None) -> AbstractTaskArgs:
    return AbstractTaskArgs(
        sensors=get_core_sensors() if core_sensors is None else core_sensors,
        action_names=ALL_STRETCH_ACTIONS,
        max_steps=max_steps,
        reward_config=None,
    )


def add_extra_sensors_to_task_args(
    task_args: AbstractTaskArgs, extra_sensors: Optional[Sequence[Sensor]]
):
    if extra_sensors is None or len(extra_sensors) == 0:
        return

    core_sensor_dict = {x.uuid: x for x in task_args["sensors"]}

    for sensor in extra_sensors:
        if sensor.uuid in core_sensor_dict:
            del core_sensor_dict[sensor.uuid]

    task_args["sensors"] = list(core_sensor_dict.values()) + list(extra_sensors)
