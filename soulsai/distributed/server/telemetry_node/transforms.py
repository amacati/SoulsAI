"""Transformation module for telemetry data transformations."""

from __future__ import annotations

import operator
import time
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Callable

from soulsai.utils import module_type_from_string

if TYPE_CHECKING:
    from tensordict import TensorDict, NestedKey

telemetry_transform: Callable[[str], type[TelemetryTransform]] = module_type_from_string(__name__)


class TelemetryTransform(ABC):
    """Base class for telemetry data transformations."""

    def __init__(self):
        """Initialize the telemetry transformation."""
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
    """Create a telemetry metric from a key in the sample dictionary."""

    def __init__(self, key: NestedKey, name: str | None = None, idx: int | None = None):
        """Initialize the transformation.

        Args:
            key: The metric key in the sample dictionary.
            name: The name of the metric.
            idx: Optional index of the value. Defaults to None.
        """
        super().__init__()
        self.key = key if isinstance(key,str) else tuple(key)
        self.name = name or key
        self.idx = idx

    def __call__(self, sample: TensorDict) -> tuple[str, float]:
        """Extract the metric from the sample.

        Args:
            sample: The sample to transform.

        Returns:
            The name and value of the extracted metric.
        """
        value = sample[self.key] if self.idx is None else sample[self.key][0, self.idx]
        return self.name, value.item()


class Timer(TelemetryTransform):
    """Timer transformation to measure the time since the start of the Transform."""

    def __init__(self):
        """Set the start time."""
        super().__init__()
        self._start = time.time()

    def __call__(self, sample: TensorDict) -> tuple[str, float]:
        """Return the time since the start of the telemetry transformation.

        Args:
            sample: Used for compatibility with the TelemetryTransform interface.

        Returns:
            The name and value of the time since the start of the telemetry transformation.
        """
        return "time", time.time() - self._start


class CompareValue(TelemetryTransform):
    """Compare a value from the sample to a given value."""

    def __init__(
        self,
        key: NestedKey,
        value: float,
        name: str | None = None,
        op: str = "gt",
        scale: float = 1.0,
        offset: float = 0.0,
    ):
        """Initialize the transformation.

        Args:
            key: The metric key in the sample dictionary.
            value: The value to compare to.
            name: The name of the metric.
            op: The comparison operator. Names are according to Python's operator module.
            scale: The scale factor for the value.
            offset: The offset for the value.
        """
        super().__init__()
        self.key = key if isinstance(key,str) else tuple(key)
        self.value = value
        self.name = name or key
        self._operator = getattr(operator, op)
        self._scale = scale
        self._offset = offset

    def __call__(self, sample: TensorDict) -> tuple[str, float]:
        """Compare the value from the sample to the given value.

        Args:
            sample: The sample to transform.

        Returns:
            The name and the result of the comparison as float.
        """
        value = sample[self.key] * self._scale + self._offset
        return self.name, float(self._operator(value, self.value).item())
