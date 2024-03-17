"""Transformation module for all data transformations."""

from __future__ import annotations

import io
import logging
from typing import Callable

import torch
import torch.nn as nn
from tensordict import TensorDict
from torch import Tensor

from soulsai.core.noise import Noise, noise_cls
from soulsai.core.scheduler import Scheduler, scheduler_cls
from soulsai.utils import module_type_from_string

logger = logging.getLogger(__name__)
transform_cls: Callable[[str], type[Transform]] = module_type_from_string(__name__)

NestedKey = str | tuple[str]


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

    def forward(
        self, x: TensorDict, keys_mapping: dict[NestedKey, NestedKey] | None = None
    ) -> TensorDict:
        """Return the input tensor unchanged.

        Args:
            x: Input tensor.
            keys_mapping: Optional dictionary that maps keys to new keys. Present for compatibility.

        Returns:
            Input tensor.
        """
        assert isinstance(x, TensorDict), f"Expected input to be a TensorDict, is {type(x)}"
        return x


class Chain(Transform):
    """Chain transformation class for chaining multiple transformations together."""

    def __init__(self, transforms: list[Transform | dict]):
        """Initialize the transformation.

        Args:
            transforms: List of transformations.
        """
        super().__init__()
        for i, tf in enumerate(transforms):
            if isinstance(tf, dict):
                transforms[i] = transform_cls(tf["type"])(**(tf.get("kwargs") or {}))
        assert all(isinstance(x, Transform) for x in transforms), "All elements must be Transforms"
        self.params["transforms"] = nn.ModuleList(transforms)

    def forward(
        self, x: TensorDict, keys_mapping: dict[NestedKey, NestedKey] | None = None
    ) -> TensorDict:
        """Apply the transformations in the chain to the input tensor.

        Args:
            x: Input tensor.
            keys_mapping: Optional dictionary that maps keys to new keys. This is useful when the
                transformation should be applied to a subset of the keys, or when the same
                parameters should be used for different keys, i.e. for 'obs' and 'next_obs'.

        Returns:
            Transformed TensorDict.
        """
        assert isinstance(x, TensorDict), f"Expected input to be a TensorDict, is {type(x)}"
        for tf in self.params["transforms"]:
            x = tf(x, keys_mapping)
        return x


class Normalize(Transform):
    """Normalization class for normalizing tensors to have zero mean and unit variance.

    Mean and standard deviation parameters are estimated from the data during the update step.
    """

    def __init__(
        self,
        keys: list[NestedKey],
        shapes: list[tuple[int]],
        indexes: list[list[int] | None] | None = None,
    ):
        """Initialize the mean, standard deviation and helper parameters.

        Args:
            keys: Keys of elements normalized by the transformation.
            shapes: Shapes of elements.
            indexes: Optional list of indexes to normalize. If None, all elements are normalized.
        """
        super().__init__()
        self._keys = [k if isinstance(k, str) else tuple(k) for k in keys]
        for key, shape in zip(keys, shapes):
            self.params[f"{key}_mean"] = nn.Parameter(torch.zeros(shape), requires_grad=False)
            self.params[f"{key}_std"] = nn.Parameter(torch.ones(shape), requires_grad=False)
            self.params[f"{key}_m2"] = nn.Parameter(torch.zeros(shape), requires_grad=False)
        self.params["count"] = nn.Parameter(torch.zeros(1, dtype=torch.int64), requires_grad=False)
        self.params["eps2"] = nn.Parameter(torch.tensor(1e-4), requires_grad=False)
        self._indexes = None
        if indexes is not None:
            self._indexes = {
                k: torch.tensor(idx) if idx is not None else slice(None)
                for k, idx in zip(self._keys, indexes)
            }

    def forward(
        self, x: TensorDict, keys_mapping: dict[NestedKey, NestedKey] | None = None
    ) -> TensorDict:
        """Normalize the input TensorDict to have zero mean and unit variance.

        Args:
            x: Sample TensorDict.
            keys_mapping: Optional dictionary that maps keys to new keys. This is useful when the
                transformation should be applied to a subset of the keys, or when the same
                parameters should be used for different keys, i.e. for 'obs' and 'next_obs'.

        Returns:
            Normalized TensorDict.
        """
        assert isinstance(x, TensorDict), f"Expected input to be a TensorDict, is {type(x)}"
        for key in self._keys:
            sample_key = key
            if keys_mapping is not None:  # Remap keys if necessary
                if key not in keys_mapping:  # Skip keys not in the remapping
                    continue
                sample_key = keys_mapping[key]
            idx = self._indexes[key]
            mean, std = self.params[f"{key}_mean"], self.params[f"{key}_std"]
            x[sample_key][idx] = (x[sample_key][idx] - mean[idx]) / std[idx]
        return x

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
        for key in self._keys:
            delta = x[key] - self.params[f"{key}_mean"]
            self.params[f"{key}_mean"] += torch.sum(delta / self.params["count"], axis=0)
            self.params[f"{key}_m2"] += torch.sum(
                delta * (x[key] - self.params[f"{key}_mean"]), axis=0
            )
            std2 = torch.maximum(
                self.params["eps2"], self.params[f"{key}_m2"] / self.params["count"]
            )
            self.params[f"{key}_std"].copy_(torch.sqrt(std2))


class NormalizeImg(Transform):
    """Image normalization transformation class for normalizing images to the range [-1, 1]."""

    def __init__(self, keys: list[NestedKey]):
        """Initialize the transformation."""
        super().__init__()
        self._keys = [k if isinstance(k, str) else tuple(k) for k in keys]

    def forward(
        self, x: TensorDict, keys_mapping: dict[NestedKey, NestedKey] | None = None
    ) -> TensorDict:
        """Normalize the input tensor to the range [-1, 1].

        Args:
            x: Sample TensorDict.
            keys_mapping: Optional dictionary that maps keys to new keys. This is useful when the
                normalization should be applied to a subset of the keys, or when the same parameters
                should be used for different keys, i.e. for 'obs' and 'next_obs'.

        Returns:
            Normalized TensorDict.
        """
        assert isinstance(x, TensorDict), f"Expected input to be a TensorDict, is {type(x)}"
        for key in self._keys:
            if keys_mapping is not None:  # Remap keys if necessary
                if key not in keys_mapping:  # Skip keys not in the remapping
                    continue
                key = keys_mapping[key]
            x[key] = (x[key] / 255.0) * 2.0 - 1.0
        return x


class Mask(Transform):
    """Mask the value tensor with -inf at the masked indices."""

    def __init__(self, key: NestedKey, mask_key: NestedKey, mask_value: float = -torch.inf):
        """Initialize the transformation.

        Args:
            key: Key of the action tensor in the input TensorDict.
            mask_key: Key of the action mask tensor in the input TensorDict.
            mask_value: Value to use for masking.
        """
        super().__init__()
        self._key = key if isinstance(key, str) else tuple(key)
        self._mask_key = mask_key if isinstance(mask_key, str) else tuple(mask_key)
        self._mask_value = mask_value

    def forward(
        self, x: TensorDict, keys_mapping: dict[NestedKey, NestedKey] | None = None
    ) -> TensorDict:
        """Mask the action value tensor with `self._mask_value` at the masked indices.

        Args:
            x: Input TensorDict.
            keys_mapping: Optional dictionary that maps keys to new keys. This is useful when the
                transformation should be applied to a subset of the keys, or when the same
                parameters should be used for different keys, i.e. for 'obs' and 'next_obs'.

        Returns:
            The masked TensorDict.
        """
        assert isinstance(x, TensorDict), f"Expected input to be a TensorDict, is {type(x)}"
        mask_key = self._mask_key if keys_mapping is None else keys_mapping[self._mask_key]
        key = self._key if keys_mapping is None else keys_mapping[self._key]
        assert not torch.all(x[self._mask_key] == 0), "All values are masked"
        x[key][~x[mask_key]] = self._mask_value
        return x


class GreedyAction(Transform):
    """Greedy action transformation class for selecting the action with the highest value."""

    def __init__(self, value_key: str, action_key: str):
        """Initialize the transformation.

        Args:
            value_key: Key of the value tensor in the input TensorDict.
            action_key: Key of the action tensor in the input TensorDict.
        """
        super().__init__()
        self._value_key = value_key
        self._action_key = action_key

    def forward(
        self, x: TensorDict, keys_mapping: dict[NestedKey, NestedKey] | None = None
    ) -> TensorDict:
        """Select the action with the highest value.

        Args:
            x: Input TensorDict.
            keys_mapping: Optional dictionary that maps keys to new keys. This is useful when the
                transformation should be applied to a subset of the keys, or when the same
                parameters should be used for different keys, i.e. for 'obs' and 'next_obs'.

        Returns:
            The input TensorDict with the action tensor set to the action with the highest value.
        """
        assert isinstance(x, TensorDict), f"Expected input to be a TensorDict, is {type(x)}"
        value_key = self._value_key if keys_mapping is None else keys_mapping[self._value_key]
        action_key = self._action_key if keys_mapping is None else keys_mapping[self._action_key]
        x[action_key] = torch.argmax(x[value_key], dim=-1)
        return x


class ExponentialAction(Transform):
    """Select an action using Boltzmann exploration with a temperature parameter."""

    def __init__(self, value_key: NestedKey, action_key: NestedKey, scheduler: Scheduler | dict):
        """Initialize the transformation.

        Args:
            value_key: Key of the value tensor in the input TensorDict.
            action_key: Key of the action tensor in the input TensorDict.
            scheduler: Temperature scheduler for the Boltzmann exploration.
        """
        super().__init__()
        self._value_key = value_key
        self._action_key = action_key
        if isinstance(scheduler, dict):
            scheduler = scheduler_cls(scheduler["type"])(**(scheduler.get("kwargs") or {}))
        self.params["scheduler"] = scheduler

    def forward(
        self, x: TensorDict, keys_mapping: dict[NestedKey, NestedKey] | None = None
    ) -> TensorDict:
        """Select an action using Boltzmann exploration with a temperature parameter.

        Args:
            x: Input TensorDict.
            keys_mapping: Optional dictionary that maps keys to new keys. This is useful when the
                transformation should be applied to a subset of the keys, or when the same
                parameters should be used for different keys, i.e. for 'obs' and 'next_obs'.

        Returns:
            The input TensorDict with the action tensor set to the action selected using Boltzmann
            exploration.
        """
        assert isinstance(x, TensorDict), f"Expected input to be a TensorDict, is {type(x)}"
        value_key = self._value_key if keys_mapping is None else keys_mapping[self._value_key]
        action_key = self._action_key if keys_mapping is None else keys_mapping[self._action_key]
        dist = torch.distributions.Categorical(logits=x[value_key] / self.params["scheduler"]())
        x[action_key] = dist.sample()
        return x

    def update(self, x: TensorDict):
        """Update the temperature parameter."""
        assert isinstance(x, TensorDict), f"Expected input to be a TensorDict, is {type(x)}"
        self.params["scheduler"].update(1)


class Choice(Transform):
    """Choice transformation class for selecting one of several transforms at random."""

    def __init__(self, key: NestedKey, transforms: list[Transform | dict], probs: list[float]):
        """Initialize the transformation.

        Args:
            key: Key(s) of the input tensor in the sample TensorDict.
            transforms: List of transformations.
            probs: List of probabilities for choosing each transformation.
        """
        super().__init__()
        self._key = key if isinstance(key, str) else tuple(key)

        for i, tf in enumerate(transforms):
            if isinstance(tf, dict):
                transforms[i] = transform_cls(tf["type"])(**(tf.get("kwargs") or {}))
        assert all(isinstance(x, Transform) for x in transforms), "All elements must be Transforms"

        self.params["transforms"] = nn.ModuleList(transforms)
        probs = torch.tensor(probs, dtype=torch.float32)
        assert torch.all(probs >= 0), "p must be non-negative"
        assert torch.isclose(probs.sum(), torch.tensor(1.0)), "p must sum to 1"
        assert len(probs) == len(transforms), "probs must have the same length as transforms"
        probs = probs / probs.sum()
        self.params["probs"] = nn.Parameter(probs, requires_grad=False)

    def forward(
        self, x: TensorDict, keys_mapping: dict[NestedKey, NestedKey] | None = None
    ) -> TensorDict:
        """Choose a transformation at random and apply it to the input tensor.

        Args:
            x: Sample TensorDict.
            keys_mapping: Optional dictionary that maps keys to new keys. This is useful when the
                transformation should be applied to a subset of the keys, or when the same
                parameters should be used for different keys, i.e. for 'obs' and 'next_obs'.

        Returns:
            Transformed TensorDict.
        """
        assert isinstance(x, TensorDict), f"Expected input to be a TensorDict, is {type(x)}"
        key = self._key if keys_mapping is None else keys_mapping[self._key]
        tf_idx = torch.multinomial(self.params["probs"], x[key].shape[0], replacement=True)
        x = torch.cat(
            [self.params["transforms"][j](x[[i]], keys_mapping) for i, j in enumerate(tf_idx)],
            dim=-1,
        )
        return x


class ScheduledChoice(Transform):
    """Scheduled choice transformation class for selecting one of several transforms at random.

    The chance of choosing a transformation is scheduled by a scheduler. The scheduler advances on
    each call to `update`. This allows us to implement behaviors like annealing exploration rates.
    """

    def __init__(
        self, key: NestedKey, transforms: list[Transform | dict], scheduler: Scheduler | dict
    ):
        """Initialize the transformation.

        Args:
            key: Key of the input tensor in the sample TensorDict.
            transforms: List of transformations.
            scheduler: `Scheduler` or dictionary with `type` and optional `kwargs` keys.
        """
        super().__init__()
        self._key = key if isinstance(key, str) else tuple(key)
        for i, tf in enumerate(transforms):
            if isinstance(tf, dict):
                transforms[i] = transform_cls(tf["type"])(**(tf.get("kwargs") or {}))
        assert all(isinstance(x, Transform) for x in transforms), "All elements must be Transforms"
        self.params["transforms"] = nn.ModuleList(transforms)

        if isinstance(scheduler, dict):
            scheduler = scheduler_cls(scheduler["type"])(**(scheduler.get("kwargs") or {}))
        self.params["scheduler"] = scheduler

    def forward(
        self, x: TensorDict, keys_mapping: dict[NestedKey, NestedKey] | None = None
    ) -> TensorDict:
        """Choose a transformation at random and apply it to the input tensor.

        Args:
            x: Sample TensorDict.
            keys_mapping: Optional dictionary that maps keys to new keys. This is useful when the
                transformation should be applied to a subset of the keys, or when the same
                parameters should be used for different keys, i.e. for 'obs' and 'next_obs'.

        Returns:
            Transformed TensorDict.
        """
        assert isinstance(x, TensorDict), f"Expected input to be a TensorDict, is {type(x)}"
        probs = self.params["scheduler"]()
        assert torch.all(probs >= 0), "p must be non-negative"
        probs /= probs.sum()
        key = self._key if keys_mapping is None else keys_mapping[self._key]
        tf_idx = torch.multinomial(probs, x[key].shape[0], replacement=True)
        x = torch.cat(
            [self.params["transforms"][j](x[[i]], keys_mapping) for i, j in enumerate(tf_idx)],
            dim=-1,
        )
        return x

    def update(self, _: TensorDict):
        """Update the transformation parameters.

        Args:
            x: Input tensor.
        """
        self.params["scheduler"].update(1)


class ReplaceWithNoise(Transform):
    """Replace the input tensor with noise."""

    def __init__(self, key: NestedKey, noise: Noise | dict):
        """Construct the noise if necessary and initialize the transformation.

        Args:
            key: Key of the input tensor in the sample TensorDict.
            noise: Noise object or dictionary with `type` and optional `kwargs` keys.
        """
        super().__init__()
        self._key = key if isinstance(key, str) else tuple(key)
        if isinstance(noise, dict):
            noise = noise_cls(noise["type"])(**(noise.get("kwargs") or {}))
        assert isinstance(noise, Noise), "noise must be a Noise object"
        self.noise = noise

    def forward(
        self, x: TensorDict, keys_mapping: dict[NestedKey, NestedKey] | None = None
    ) -> Tensor:
        """Replace the input tensor with noise.

        Args:
            x: Sample TensorDict.
            keys_mapping: Optional dictionary that maps keys to new keys. This is useful when the
                transformation should be applied to a subset of the keys, or when the same
                parameters should be used for different keys, i.e. for 'obs' and 'next_obs'.

        Returns:
            Transformed TensorDict.
        """
        assert isinstance(x, TensorDict), f"Expected input to be a TensorDict, is {type(x)}"
        key = self._key if keys_mapping is None else keys_mapping[self._key]
        x[key] = self.noise(x[key])
        return x
