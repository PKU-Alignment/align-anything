import sys
from collections import defaultdict
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

try:
    from typing import Literal, TypedDict
except ImportError:
    from typing_extensions import Literal, TypedDict

from attrs import define

from allenact.base_abstractions.sensor import Sensor


class Vector3(TypedDict):
    x: float
    y: float
    z: float


class KeyedDefaultDict(defaultdict):
    """
    A defaultdict that passes the key to the default_factory.
    """

    def __missing__(self, key: Any):
        return self.default_factory(key)


@define
class RewardConfig:
    step_penalty: float
    goal_success_reward: float
    failed_stop_reward: float
    shaping_weight: float
    reached_horizon_reward: float
    positive_only_reward: bool
    failed_action_penalty: float = 0.0


class AgentPose(TypedDict):
    position: Vector3
    rotation: Vector3
    horizon: int
    standing: bool


class AbstractTaskArgs(TypedDict):
    sensors: List[Sensor]
    max_steps: int
    action_names: List[str]
    reward_config: Optional[RewardConfig]


class THORActions:
    move_ahead = "m"
    move_back = "b"
    rotate_right = "r"
    rotate_left = "l"
    rotate_right_small = "rs"
    rotate_left_small = "ls"
    done = "end"
    move_arm_up = "yp"
    move_arm_up_small = "yps"
    move_arm_down = "ym"
    move_arm_down_small = "yms"
    move_arm_out = "zp"
    move_arm_out_small = "zps"
    move_arm_in = "zm"
    move_arm_in_small = "zms"
    wrist_open = "wp"
    wrist_close = "wm"
    pickup = "p"
    dropoff = "d"
    ARM_ACTIONS = [
        move_arm_in,
        move_arm_out,
        move_arm_up,
        move_arm_down,
        move_arm_in_small,
        move_arm_out_small,
        move_arm_up_small,
        move_arm_down_small,
    ]
    MOVE_ACTIONS = [
        move_ahead,
        move_back,
    ]
    ROTATE_ACTIONS = [
        rotate_right,
        rotate_left,
        rotate_right_small,
        rotate_left_small,
    ]
    sub_done = "sub_done"

    @classmethod
    def get_action_name(cls, short_string):
        for name, value in cls.__dict__.items():
            if value == short_string:
                return name
        return None


REGISTERED_TASK_PARAMS: Dict[str, List[str]] = {}

if sys.version_info >= (3, 9):

    def get_required_keys(cls):
        return getattr(cls, "__required_keys__", [])

else:

    def get_required_keys(cls):
        return list(cls.__annotations__.keys())


def register_task_specific_params(cls):
    REGISTERED_TASK_PARAMS[cls.__name__] = get_required_keys(cls)
    return cls


class ObjectInstr(TypedDict):
    synsets: List[str]


class ObjectEval(TypedDict):
    synset_to_object_ids: Dict[str, List[str]]
    broad_synset_to_object_ids: Dict[str, List[str]]


class ObjectNav(ObjectInstr, ObjectEval):
    pass


class Fetch(ObjectInstr, ObjectEval):
    pass


class ObjRoom(TypedDict):
    room_type: str


class RequiresVisits(TypedDict):
    visit_ids: Dict[str, List[str]]


class RelAttribute(RequiresVisits, ObjRoom):
    rel_attribute: Union[str, Tuple[str, str]]


class LocalRef(RequiresVisits):
    reference_type: str
    reference_synsets: List[str]


class Affordance(TypedDict):
    affordance: str


class OpenDescription(TypedDict):
    uid: str


@register_task_specific_params
class ObjectNavType(ObjectNav):
    pass


@register_task_specific_params
class EasyObjectNavType(ObjectNav):
    pass


@register_task_specific_params
class ObjectNavRoom(ObjectNav, ObjRoom):
    pass


@register_task_specific_params
class ObjectNavRelAttribute(ObjectNav, RelAttribute):
    pass


@register_task_specific_params
class ObjectNavAffordance(ObjectNav, Affordance):
    pass


@register_task_specific_params
class ObjectNavLocalRef(ObjectNav, LocalRef):
    pass


@register_task_specific_params
class ObjectNavDescription(ObjectNav, OpenDescription):
    pass


@register_task_specific_params
class ObjectNavMulti(ObjectNav):
    pass


@register_task_specific_params
class BPEObjectNavType(ObjectNav):
    pass


@register_task_specific_params
class BPEObjectNavMulti(ObjectNav):
    pass


@register_task_specific_params
class FetchType(Fetch):
    pass


@register_task_specific_params
class EasyFetchType(Fetch):
    pass


@register_task_specific_params
class PickupType(Fetch):
    pass


@register_task_specific_params
class RoomNav(TypedDict):
    room_types: List[str]
    room_ids: Dict[str, List[str]]


@register_task_specific_params
class RoomVisit(TypedDict):
    num_rooms_in_house: int


@register_task_specific_params
class GoToPoint(TypedDict):
    location_type: str
    goal_in_camera_2d_first_step: Tuple[float, float]
    goal_in_world_3d: Dict[str, float]
    pass


@register_task_specific_params
class GoNearPoint(TypedDict):
    location_type: str
    target_obj_in_3d: Dict[str, float]
    possible_points_on_target_in_first_frame: List[Tuple[float, float]]
    object_type: str
    object_id: str
    pass


def get_task_relevant_synsets(task_spec: Dict[str, Any]) -> List[str]:
    """Given an input task spec, returns a list of all synsets relevant to that task's success."""
    synsets = set()
    for k, v in task_spec.items():
        if "synset" in k:
            if k.endswith("synset_to_object_ids"):
                assert isinstance(v, Dict)
                synsets.update(v.keys())
            elif k in ["synsets", "reference_synsets"]:
                assert isinstance(v, Sequence)
                synsets.update(v)
            else:
                raise NotImplementedError
    return list(synsets)
