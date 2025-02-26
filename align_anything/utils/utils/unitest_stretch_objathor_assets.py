import os
import pdb
import random
import time
import prior
import ai2thor
import ai2thor.platform
from ai2thor.controller import Controller
from ai2thor.hooks.procedural_asset_hook import (
    ProceduralAssetHookRunner,
    get_all_asset_ids_recursively,
    create_assets_if_not_exist,
)
from environment.stretch_controller import StretchController
from utils.type_utils import THORActions
from online_evaluation.online_evaluation_types_and_utils import (
    NormalizedEvalSample,
    EvalSample,
    eval_sample_to_normalized_eval_sample,
)
from tasks.multi_task_eval_sampler import MultiTaskSampler

ACTION = [
    "MoveAhead",
    "RotateRight",
    "RotateLeft",
]

ACTIONStretch = [
    THORActions.move_ahead,
    THORActions.move_back,
    THORActions.rotate_right,
    THORActions.rotate_left,
    THORActions.rotate_right_small,
    THORActions.rotate_left_small,
]


class ProceduralAssetHookRunnerResetOnNewHouse(ProceduralAssetHookRunner):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.last_asset_id_set = set()

    def Initialize(self, action, controller):
        if self.asset_limit > 0:
            return controller.step(
                action="DeleteLRUFromProceduralCache", assetLimit=self.asset_limit
            )

    def CreateHouse(self, action, controller):
        house = action["house"]
        asset_ids = get_all_asset_ids_recursively(house["objects"], [])
        asset_ids_set = set(asset_ids)
        if not asset_ids_set.issubset(self.last_asset_id_set):
            controller.step(action="DeleteLRUFromProceduralCache", assetLimit=0)
            self.last_asset_id_set = set(asset_ids)

        return create_assets_if_not_exist(
            controller=controller,
            asset_ids=asset_ids,
            asset_directory=self.asset_directory,
            asset_symlink=self.asset_symlink,
            stop_if_fail=self.stop_if_fail,
            copy_to_dir=os.path.join(controller._build.base_dir, self.target_dir),
            load_file_in_unity=False,
        )


def export_objaverse_assets():
    import os
    import warnings

    ASSETS_VERSION = "2023_07_28"
    OBJAVERSE_DATA_DIR = os.path.abspath(
        os.environ.get("OBJAVERSE_DATA_DIR", os.path.expanduser("~/.objathor-assets"))
    )

    if not os.path.basename(OBJAVERSE_DATA_DIR) == ASSETS_VERSION:
        OBJAVERSE_DATA_DIR = os.path.join(OBJAVERSE_DATA_DIR, ASSETS_VERSION)

    OBJAVERSE_ASSETS_DIR = os.environ.get(
        "OBJAVERSE_ASSETS_DIR", os.path.join(OBJAVERSE_DATA_DIR, "assets")
    )
    OBJAVERSE_HOUSES_DIR = os.environ.get("OBJAVERSE_HOUSES_DIR")

    for var_name in ["OBJAVERSE_ASSETS_DIR", "OBJAVERSE_HOUSES_DIR"]:
        if locals()[var_name] is None:
            warnings.warn(f"{var_name} is not set.")
        else:
            locals()[var_name] = os.path.abspath(locals()[var_name])

    if OBJAVERSE_HOUSES_DIR is None:
        warnings.warn("`OBJAVERSE_HOUSES_DIR` is not set.")
    else:
        OBJAVERSE_HOUSES_DIR = os.path.abspath(OBJAVERSE_HOUSES_DIR)

    print(
        f"Using"
        f" '{OBJAVERSE_ASSETS_DIR}' for objaverse assets,"
        f" '{OBJAVERSE_HOUSES_DIR}' for procthor-objaverse houses."
    )
    return OBJAVERSE_ASSETS_DIR, OBJAVERSE_HOUSES_DIR


def load_minival_eval_samples_per_task(task_type: str):
    EVAL_TASKS = prior.load_dataset(
        dataset="vida-benchmark",
        revision="chores-small",
        task_types=["ObjectNavType"],
    )

    samples = EVAL_TASKS["val"]

    sample_ids = list(range(len(samples)))

    normalized_samples = [
        eval_sample_to_normalized_eval_sample(task_type=task_type, sample=samples[i], index=i)
        for i in range(len(samples))
    ]

    return [normalized_samples[i] for i in sample_ids]


def test_objathor():
    on_server = bool(int(os.environ.get("ON_SERVER", False)))

    OBJAVERSE_ASSETS_DIR, OBJAVERSE_HOUSES_DIR = export_objaverse_assets()
    _ACTION_HOOK_RUNNER = ProceduralAssetHookRunnerResetOnNewHouse(
        asset_directory=OBJAVERSE_ASSETS_DIR, asset_symlink=True, verbose=True, asset_limit=200
    )

    max_houses_per_split = {"train": 64, "val": 0, "test": 0}

    house = prior.load_dataset(
        dataset="spoc-data",
        entity="spoc-robot",
        revision="local-objaverse-procthor-houses",
        path_to_splits=None,
        split_to_path={
            k: os.path.join(OBJAVERSE_HOUSES_DIR, f"{k}.jsonl.gz") for k in ["train", "val", "test"]
        },
        max_houses_per_split=max_houses_per_split,
    )

    # STRETCH_BUILD_ID = "5e43486351ac6339c399c199e601c9dd18daecc3"
    STRETCH_BUILD_ID = "ca27b6ffa6881de3711e3b8cce184c8b1111a325"
    if on_server:
        device = 0
    else:
        device = None
    STRETCH_ENV_ARGS = dict(
        gridSize=0.2 * 0.75,
        width=396,
        height=224,
        visibilityDistance=1.0,
        fieldOfView=59,
        server_class=ai2thor.fifo_server.FifoServer,
        useMassThreshold=True,
        massThreshold=10,
        autoSimulation=False,
        autoSyncTransforms=True,
        agentMode="stretch",
        renderInstanceSegmentation=True,
        renderDepthImage=True,
        cameraNearPlane=0.01,
        commit_id=STRETCH_BUILD_ID,
        server_timeout=10000,
        snapToGrid=False,
        fastActionEmit=True,
        action_hook_runner=_ACTION_HOOK_RUNNER,
        platform=ai2thor.platform.CloudRendering if on_server else None,
        gpu_device=device,
    )
    c = StretchController(**STRETCH_ENV_ARGS)
    c.reset(scene=house["train"][0])
    c.controller.step("Pass")

    EVAL_TASKS = load_minival_eval_samples_per_task("ObjectNavType")

    import pdb

    pdb.set_trace()
    t_start = time.time()
    for i in range(100):
        a = random.choice(ACTIONStretch)
        c.agent_step(a)
        # c.controller.step("Pass")
    t_end = time.time()
    print(f"100 steps, estimated time in python {t_end - t_start}")


if __name__ == "__main__":
    test_objathor()
