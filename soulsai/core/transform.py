"""Transformation module for all data transformations."""
from __future__ import annotations

from typing import Callable, Mapping
import io
import logging

import torch
from torch import Tensor
import torch.nn as nn
from tensordict import TensorDict

from soulsai.core.noise import noise_cls, Noise
from soulsai.core.scheduler import scheduler_cls, Scheduler
from soulsai.utils import module_type_from_string

logger = logging.getLogger(__name__)
transform_cls: Callable[[str], type[Transform]] = module_type_from_string(__name__)


class Transform(nn.Module):
    """Base class for transformations."""

    def __init__(self):
        """Initialize the transformation."""
        super().__init__()
        self.params = nn.ParameterDict()

    def update(self, x: TensorDict):
        """Update the transformation parameters.

        The base update is a no-op so that only transformations that require updates need to
        implement this method.

        Args:
            x: Input tensor.
        """
        ...

    def serialize(self) -> bytes:
        """Serialize the transformation into bytes for synchronization across nodes."""
        buff = io.BytesIO()
        torch.save(self.state_dict(), buff)
        buff.seek(0)
        return buff.read()

    def deserialize(self, serialization: bytes):
        """Deserialize the transformation from bytes and load them into the transformation."""
        buff = io.BytesIO(serialization)
        buff.seek(0)
        self.load_state_dict(torch.load(buff))


class Identity(Transform):
    """Identity transformation class for no transformation."""

    def forward(self, x: TensorDict) -> TensorDict:
        """Return the input tensor unchanged.

        Args:
            x: Input tensor.

        Returns:
            Input tensor.
        """
        return x


class TensorNormalization(Transform):
    """Standard normalization transformation class for Tensors.

    The normalization is done using Welford's algorithm for numerical stability.
    """

    def __init__(self, key: str, shape: tuple[int]):
        """Initialize the mean, standard deviation and helper parameters.

        Args:
            key: Key of the input tensor in update TensorDict.
            shape: Shape of the input tensor.
        """
        super().__init__()
        self.key = key
        self.params["mean"] = nn.Parameter(torch.zeros(shape), requires_grad=False)
        self.params["std"] = nn.Parameter(torch.ones(shape), requires_grad=False)
        self.params["m2"] = nn.Parameter(torch.zeros(shape), requires_grad=False)
        self.params["count"] = nn.Parameter(torch.zeros(1, dtype=torch.int64), requires_grad=False)
        self.params["eps2"] = nn.Parameter(torch.tensor(1e-4), requires_grad=False)

    def forward(self, x: Tensor) -> tuple[Tensor]:
        """Normalize the input tensor to have zero mean and unit variance.

        Args:
            x: Input tensor.

        Returns:
            Normalized tensor.
        """
        assert isinstance(x, Tensor), f"Expected input to be a Tensor, is {type(x)}"
        return (x - self.params["mean"]) / self.params["std"]

    def update(self, x: TensorDict):
        """Update the mean and standard deviation estimate from a sample TensorDict batch.

        Args:
            x: Input TensorDict containing the tensor at `self.key`.
        """
        assert isinstance(x, TensorDict), f"Expected input to be a TensorDict, is {type(x)}"
        x = x[self.key]
        assert isinstance(x, Tensor), f"Expected update value to be a Tensor, is {type(x)}"
        # A batched variant of Welford's algorithm
        # See https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Welford's_online_algorithm  # noqa: E501
        self.params["count"] += x.shape[0]
        delta = x - self.params["mean"]
        self.params["mean"] += torch.sum(delta / self.params["count"], axis=0)
        self.params["m2"] += torch.sum(delta * (x - self.params["mean"]), axis=0)
        # Numerical stability
        std2 = torch.maximum(self.params["eps2"], self.params["m2"] / self.params["count"])
        self.params["std"][:] = torch.sqrt(std2)


class TensorDictNormalization(Transform):
    """Standard normalization transformation class for TensorDicts."""

    def __init__(self, shapes: Mapping[str, tuple[int]]):
        """Initialize the mean, standard deviation and helper parameters.

        Args:
            shapes: Dictionary of keys and shapes of the input tensors in the TensorDict. If nested,
                the keys in the TensorDict are concatenated with a dot. Keys in the shapes
                dictionary must match the keys in the TensorDict accordingly.
        """
        super().__init__()
        for key, shape in shapes.items():
            self.params[f"{key}_mean"] = nn.Parameter(torch.zeros(shape), requires_grad=False)
            self.params[f"{key}_std"] = nn.Parameter(torch.ones(shape), requires_grad=False)
            self.params[f"{key}_m2"] = nn.Parameter(torch.zeros(shape), requires_grad=False)
        self.params["count"] = nn.Parameter(torch.zeros(1, dtype=torch.int64), requires_grad=False)
        self.params["eps2"] = nn.Parameter(torch.tensor(1e-4), requires_grad=False)

    def forward(self, x: TensorDict) -> Tensor:
        """Normalize the input TensorDict to have zero mean and unit variance.

        Args:
            x: Input TensorDict.

        Returns:
            Normalized TensorDict.
        """
        assert isinstance(x, TensorDict), f"Expected input to be a TensorDict, is {type(x)}"
        norm_td = TensorDict(
            {
                k: (v - self.params[f"{k}_mean"]) / self.params[f"{k}_std"]
                for k, v in x.flatten_keys().items()
            },
            batch_size=x.batch_size,
            device=x.device).unflatten_keys()
        return norm_td

    def update(self, x: TensorDict):
        """Update the keys' mean and standard deviation estimate from a sample TensorDict batch.

        Args:
            x: Sample TensorDict.
        """
        assert isinstance(x, TensorDict), f"Expected input to be a TensorDict, is {type(x)}"
        assert len(x.batch_size) == 1, f"Batch size must be a scalar, is {x.batch_size}"
        self.params["count"] += x.batch_size[0]
        # A batched variant of Welford's algorithm
        # See https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Welford's_online_algorithm  # noqa: E501
        for key, value in x.flatten_keys().items():
            delta = value - self.params[f"{key}_mean"]
            self.params[f"{key}_mean"] += torch.sum(delta / self.params["count"], axis=0)
            self.params[f"{key}_m2"] += torch.sum(delta * (value - self.params[f"{key}_mean"]),
                                                  axis=0)
            std2 = torch.maximum(self.params["eps2"],
                                 self.params[f"{key}_m2"] / self.params["count"])
            self.params[f"{key}_std"].copy_(torch.sqrt(std2))


class Choice(Transform):
    """Choice transformation class for selecting one of several transforms at random."""

    def __init__(self, transforms: list[Transform | dict], prob: list[float]):
        """Initialize the transformation.

        Args:
            transforms: List of transformations.
            prob: List of probabilities for choosing each transformation.
        """
        super().__init__()
        for i, tf in enumerate(transforms):
            if isinstance(tf, dict):
                transforms[i] = transform_cls(tf["type"])(**(tf.get("kwargs") or {}))
        assert all(isinstance(x, Transform) for x in transforms), "All elements must be Transforms"

        self.params["transforms"] = nn.ModuleList(transforms)
        prob = torch.tensor(prob, dtype=torch.float32)
        assert torch.all(prob >= 0), "p must be non-negative"
        assert torch.isclose(prob.sum(), torch.tensor(1.0)), "p must sum to 1"
        assert len(prob) == len(transforms), "prob must have the same length as transforms"
        prob = prob / prob.sum()
        self.params["prob"] = nn.Parameter(prob, requires_grad=False)

    def forward(self, x: Tensor) -> Tensor:
        """Choose a transformation at random and apply it to the input tensor.

        Args:
            x: Input tensor.

        Returns:
            Transformed tensor.
        """
        tf_idx = torch.multinomial(self.params["prob"], x.shape[0], replacement=True)
        return torch.stack([self.params["transforms"][j](x[i])[0] for i, j in enumerate(tf_idx)])


class ScheduledChoice(Transform):

    def __init__(self, transforms: list[Transform | dict], scheduler: Scheduler | dict):
        """Initialize the transformation.

        Args:
            transforms: List of transformations.
            prob: List of probabilities for choosing each transformation.
        """
        super().__init__()
        for i, tf in enumerate(transforms):
            if isinstance(tf, dict):
                transforms[i] = transform_cls(tf["type"])(**(tf.get("kwargs") or {}))
        assert all(isinstance(x, Transform) for x in transforms), "All elements must be Transforms"
        self.params["transforms"] = nn.ModuleList(transforms)

        if isinstance(scheduler, dict):
            scheduler = scheduler_cls(scheduler["type"])(**(scheduler.get("kwargs") or {}))
        self.params["scheduler"] = scheduler

    def forward(self, x: Tensor) -> Tensor:
        """Choose a transformation at random and apply it to the input tensor.

        Args:
            x: Input tensor.

        Returns:
            Transformed tensor.
        """
        probs = self.params["scheduler"]()
        assert torch.all(probs >= 0), "p must be non-negative"
        probs /= probs.sum()
        tf_idx = torch.multinomial(probs, x.shape[0], replacement=True)
        return torch.stack([self.params["transforms"][j](x[i])[0] for i, j in enumerate(tf_idx)])

    def update(self, x: TensorDict):
        """Update the transformation parameters.

        Args:
            x: Input tensor.
        """
        self.params["scheduler"].update(1)


class ReplaceWithNoise(Transform):
    """Replace the input tensor with noise."""

    def __init__(self, noise: Noise | dict):
        """Construct the noise if necessary and initialize the transformation.

        Args:
            noise: Noise object or dictionary with `type` and optional `kwargs` keys.
        """
        super().__init__()
        if isinstance(noise, dict):
            noise = noise_cls(noise["type"])(**(noise.get("kwargs") or {}))
        assert isinstance(noise, Noise), "noise must be a Noise object"
        self.noise = noise

    def forward(self, x: Tensor) -> Tensor:
        """Replace the input tensor with noise."""
        return self.noise(x)
