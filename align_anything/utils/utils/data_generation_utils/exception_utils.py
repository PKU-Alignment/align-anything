# define Python user-defined exceptions


class TaskSamplerException(Exception):
    """Task Sampler failed to find a valid sample"""

    pass


class HouseInvalidForTaskException(TaskSamplerException):
    """Task Sampler failed to find a valid sample because the house was fully impossible to generate any task for"""

    pass


class TaskSamplerInInvalidStateError(TaskSamplerException):
    """Task sampler has entered some totally invalid state from which next_task calls will definitely fail."""

    pass
