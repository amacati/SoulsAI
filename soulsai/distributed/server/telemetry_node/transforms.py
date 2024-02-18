from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable, TYPE_CHECKING
import operator

from soulsai.utils import module_type_from_string

if TYPE_CHECKING:
    from tensordict import TensorDict

telemetry_transform: Callable[[str], type[TelemetryTransform]] = module_type_from_string(__name__)


class TelemetryTransform(ABC):

    def __init__(self):
        super().__init__()

    @abstractmethod
    def __call__(self, sample: TensorDict) -> tuple[str, float]:
        """Transform the sample into a tuple of name and value.

        Args:
            sample: The sample to transform.

        Returns:
            A tuple of the name and value of the sample.
        """
        ...


class MetricByKey(TelemetryTransform):

    def __init__(self, key: str, name: str | None = None):
        super().__init__()
        self.key = key
        self.name = name or key

    def __call__(self, sample: TensorDict) -> tuple[str, float]:
        return self.name, sample[self.key]


class CompareValue(TelemetryTransform):

    def __init__(self,
                 key: str,
                 value: float,
                 name: str | None = None,
                 op: str = "gt",
                 scale: float = 1.0,
                 offset: float = 0.0):
        super().__init__()
        self.key = key
        self.value = value
        self.name = name or key
        self._operator = getattr(operator, op)
        self._scale = scale
        self._offset = offset

    def __call__(self, sample: TensorDict) -> tuple[str, float]:
        value = sample[self.key] * self._scale + self._offset
        return self.name, self._operator(value, self.value)
