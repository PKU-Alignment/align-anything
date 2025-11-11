# Copyright 2024 Allen Institute for AI
# ==============================================================================

from typing import Any, Dict, List, Literal, Optional, Type, Union

from allenact.utils.misc_utils import prepare_locals_for_super
from allenact.utils.system import get_logger

from eval_anything.third_party.SPOC.tasks import REGISTERED_TASKS
from eval_anything.third_party.SPOC.tasks.abstract_task import AbstractSafeTask
from eval_anything.third_party.SPOC.tasks.abstract_task_sampler import AbstractSafeTaskSampler
from eval_anything.third_party.SPOC.tasks.object_nav_task import ObjectNavTask
from eval_anything.third_party.SPOC.tasks.task_specs import (
    TaskSpec,
    TaskSpecDataset,
    TaskSpecSampler,
    TaskSpecSamplerDatasetWrapper,
)
from eval_anything.third_party.SPOC.utils.constants.stretch_initialization_utils import HORIZON
from eval_anything.third_party.SPOC.utils.type_utils import (
    REGISTERED_TASK_PARAMS,
    AbstractTaskArgs,
    AgentPose,
    Vector3,
)


class MultiTaskSampler(AbstractSafeTaskSampler):
    def __init__(
        self,
        mode: Literal['train', 'val', 'test'],
        task_args: AbstractTaskArgs,
        houses: List[Dict[str, Any]],
        house_inds: List[int],
        controller_args: Dict[str, Any],
        controller_type: Type,
        task_spec_sampler: Union[TaskSpecSampler, TaskSpecDataset],
        visualize: bool,
        prob_randomize_materials: float = 0,
        task_type: Optional[Type] = None,
        device: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**prepare_locals_for_super(locals()))

        assert (
            task_type is None
            and kwargs.get('max_tasks') is None
            and kwargs.get('sample_per_house') is None
        ), (
            'MultiTaskSampler does not support the `task_type`, `max_tasks`, or `sample_per_house` arguments.'
            ' You can control these parameters via the task_spec_sampler.'
        )

        self.mode = mode.strip().lower()
        assert self.mode in ['train', 'val', 'test']

        if isinstance(task_spec_sampler, TaskSpecDataset):
            task_spec_sampler = TaskSpecSamplerDatasetWrapper(task_spec_dataset=task_spec_sampler)

        self.task_spec_sampler = task_spec_sampler

        self.visualize = visualize

        assert self.mode == 'train' or self.prob_randomize_materials == 0

    @property
    def current_task_spec(self) -> TaskSpec:
        return self.task_spec_sampler.last_task_spec

    @property
    def length(self) -> Union[int, float]:
        """Length.
        # Returns
        Number of total tasks remaining that can be sampled. Can be float('inf').
        """
        return self.task_spec_sampler.num_remaining()

    @property
    def total_unique(self) -> Optional[Union[int, float]]:
        return len(self.task_spec_sampler)

    @property
    def current_house_index(self) -> int:
        task_spec = self.current_task_spec
        return task_spec.get('house_index')

    @staticmethod
    def task_spec_to_task_info(
        task_spec: TaskSpec, house_index: int, house: Dict[str, Any]
    ) -> Dict[str, Any]:
        agent_starting_position = {
            key: value for key, value in zip(['x', 'y', 'z'], task_spec['agent_starting_position'])
        }

        task_info = {
            'task_type': task_spec['task_type'],
            'house_index': str(house_index),
            'num_rooms': len(house['rooms']),
            'agent_starting_position': agent_starting_position,
            'agent_y_rotation': task_spec['agent_y_rotation'],
            'natural_language_spec': task_spec['natural_language_spec'],
        }

        if 'eval_info' in task_spec:
            task_info['eval_info'] = task_spec['eval_info']

        for key in REGISTERED_TASK_PARAMS.get(task_spec['task_type']):
            if key in task_spec:
                task_info[key] = task_spec[key]

        # Handle the remaining keys not found in current_task
        remaining_keys = set(REGISTERED_TASK_PARAMS.get(task_spec['task_type'])) - set(
            task_spec.keys()
        )

        if len(remaining_keys) > 0:
            raise NotImplementedError(
                f'Some keys require by the task are missing. You have given {task_info.keys()} but we require'
                f" {REGISTERED_TASK_PARAMS.get(task_spec['task_type'])}, i.e. {remaining_keys} are missing."
                f' Ping Jordi if this is surprising.'
            )

        return task_info

    def get_current_task_spec_task_info(self):  # populate eval task info
        return self.task_spec_to_task_info(
            task_spec=self.current_task_spec,
            house_index=self.current_house_index,
            house=self.house_index_to_house[self.current_house_index],
        )

    def increment_task_and_reset_house(
        self, force_advance_scene: bool, house_index: Optional[int] = None
    ):
        # Now self.current_task_spec will be the next task spec
        last_task_spec = self.current_task_spec
        new_task_spec = self.task_spec_sampler.next_task_spec(
            force_advance_scene=force_advance_scene, house_index=house_index
        )

        if last_task_spec is None:
            last_task_spec = {'house_index': -1, 'task_type': ''}

        house_changed = last_task_spec['house_index'] != new_task_spec['house_index']

        nav_only_tasks = [
            t.task_type_str
            for t in [
                ObjectNavTask,
            ]
        ]
        tasks_are_nav_only = (
            last_task_spec['task_type'] in nav_only_tasks
            and new_task_spec['task_type'] in nav_only_tasks
        )

        # The above code ensure that self.current_house will now be the next house
        self.reset_controller_in_current_house_and_cache_house_data(
            skip_controller_reset=self.mode == 'train'
            and (not house_changed)
            and tasks_are_nav_only,
        )

    def next_task(
        self,
        force_advance_scene: bool = False,
        house_index: Optional[int] = None,
    ) -> Optional[AbstractSafeTask]:
        # NOTE: Stopping condition
        if self.length == 0:
            return None

        self.increment_task_and_reset_house(
            force_advance_scene=force_advance_scene, house_index=house_index
        )
        assert house_index is None or self.current_house_index == house_index

        task_info = self.get_current_task_spec_task_info()
        task_info['extras'] = {}

        # Enforce the task_info's house index
        house_index = int(task_info['house_index'])
        if house_index != self.current_house_index:
            raise RuntimeError(
                f'House index does not match! {house_index} != {self.current_house_index}'
            )

        starting_pose = AgentPose(
            position=task_info['agent_starting_position'],
            rotation=Vector3(x=0, y=task_info['agent_y_rotation'], z=0),
            horizon=HORIZON,
            standing=True,
        )

        # event = self.controller.step(
        #     action="TeleportFull",
        #     **starting_pose,
        # )
        try:
            event = self.controller.teleport_agent(
                **starting_pose,
            )
        except TimeoutError:
            self.allocate_a_new_stretch_controller(use_original_ai2thor_controller=False)
            self.reset_controller_in_current_house_and_cache_house_data(skip_controller_reset=False)
            return self.next_task(force_advance_scene=force_advance_scene, house_index=house_index)
        if not event:
            if self.mode == 'train':
                self.controller.reset(self.current_house)
                event = self.controller.teleport_agent(
                    **starting_pose,
                )
                self.controller.calibrate_agent()
                if not event:
                    get_logger().warning(
                        f'Teleport failing in {self.current_house_index} at {starting_pose}'
                    )
                    get_logger().warning(event)
                    return self.next_task(
                        force_advance_scene=force_advance_scene, house_index=house_index
                    )
            else:
                # This **must** be an error during eval rather than a warning for fairness/consistency.
                raise RuntimeError(
                    f"Teleport failed in {self.current_house_index} at {task_info['starting_pose']}"
                )

        self._last_sampled_task = REGISTERED_TASKS.get(task_info['task_type'])(
            controller=self.controller,
            task_info=task_info,
            **self.task_args,
            house=self.house_index_to_house[self.current_house_index],
            visualize=self.visualize,
        )
        return self._last_sampled_task

    def reset(self):
        self.task_spec_sampler.reset()
