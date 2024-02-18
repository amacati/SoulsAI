"""The normalizer module implements a zero mean, unit variance normalizer.

The normalization is ensured to be numerically stable by setting a lower bound on the possible
standard deviation of values.
"""
from __future__ import annotations
import logging
from typing import List, Any, Type
import io
from abc import ABC, abstractmethod
from typing import Callable

import numpy as np
import torch
import torch.nn as nn

from soulsai.utils import module_type_from_string

logger = logging.getLogger(__name__)

normalizer_cls: Callable[[str], Type[AbstractNormalizer]] = module_type_from_string(__name__)


class AbstractNormalizer(nn.Module, ABC):

    def __init__(self):
        super().__init__()
        self.norm_params = nn.ParameterDict()

    @abstractmethod
    def normalize(self, x: List | np.ndarray | torch.Tensor) -> torch.Tensor:
        """Normalize the input data.

        Args:
            x: Input data array.

        Returns:
            The normalized data.
        """

    @abstractmethod
    def update(self, x: List | np.ndarray | torch.Tensor):
        """Update the normalizer parameters with the values in ``x``.

        Args:
            x: Batch of observations.
        """

    def serialize(self) -> dict:
        """Serialize the normalizer by dumping the parameters in the norm parameter dictionary.

        Returns:
            The dictionary containing the saved parameters.
        """
        param_buff = io.BytesIO()
        torch.save(self.norm_params, param_buff)
        param_buff.seek(0)
        return {"norm_params": param_buff.read()}

    @staticmethod
    def deserialize(serialization: dict) -> nn.ParameterDict:
        """Deserialize the norm parameter buffers in the state dict.

        Args:
            serialization: Dictionary containing the byte objects of the normalizer's parameters.
        """
        param_buff = io.BytesIO(serialization["norm_params"])
        param_buff.seek(0)
        params = torch.load(param_buff)
        return params

    def load_params(self, norm_params: nn.ParameterDict):
        """Load the parameter tensors into the normalizer.

        Warning:
            Parameter loading is intended to update the client normalizers and only updates the
            parameters required for normalizing. Does `NOT` update non-parameter buffers.

        Args:
            norm_params: The normalizer parameters.
        """
        # If the normalizer parameters are in shared memory, overwriting them will not change the
        # data in other processes. Therefore, we need to copy the new parameters into the existing
        # shared buffers
        for name, param in norm_params.items():
            self.norm_params[name].copy_(param)

    @staticmethod
    def _sanitize_input(x: List | np.ndarray | torch.Tensor) -> torch.Tensor:
        if isinstance(x, torch.Tensor):
            return x.float()
        if isinstance(x, np.ndarray):
            return torch.from_numpy(x).float()
        if isinstance(x, List):
            return torch.tensor(x).float()
        raise TypeError(f"Unsupported input type {x.__class__.__name__} for normalizer")


class Normalizer(AbstractNormalizer):
    """Normalizer class for preprocessing on both the client and the server side.

    Normalizes tensors to zero mean, unit variance by updating its statistics over previously seen
    values. Also includes functions to serialize the necessary parameters (e.g. server side),
    deserialize parameters (e.g. client side), and load them into the normalizer to complete an
    update.
    """

    def __init__(self,
                 obs_shape: tuple[int, ...],
                 eps: float = 1e-2,
                 clip: float = np.inf,
                 idx_list: List | None = None):
        """Initialize the normalizer parameters.

        Args:
            obs_shape: Observation shape.
            eps: Minimum denominator for normalization. Enforces stability in case of low variances.
            clip: Normalization clipping value. Restricts normalized values to the interval of
                [-clip, clip].
            idx_list: List of indices to include in the normalization. If not provided, all indices
                are normalized.
        """
        super().__init__()
        self.obs_shape = obs_shape
        self.obs_dim = len(obs_shape)
        self.clip = clip
        mask = torch.ones(obs_shape, dtype=torch.bool)
        if idx_list is not None:
            mask[:] = False
            mask[idx_list] = True
        self.norm_params = nn.ParameterDict({
            "mask": nn.Parameter(mask, requires_grad=False),
            "mean": nn.Parameter(torch.zeros(obs_shape, dtype=torch.float32), requires_grad=False),
            "std": nn.Parameter(torch.ones(obs_shape, dtype=torch.float32), requires_grad=False),
        })
        self.eps2 = torch.ones(obs_shape, dtype=torch.float32) * eps**2
        self.count = nn.Parameter(torch.tensor(0, dtype=torch.int64), requires_grad=False)
        self._m2 = nn.Parameter(torch.zeros(obs_shape, dtype=torch.float32), requires_grad=False)

    def normalize(self, x: List | np.ndarray | torch.Tensor) -> torch.Tensor:
        """Normalize the input data with the current mean and variance estimate.

        Args:
            x: Input data array.
        Returns:
            The normalized data.
        """
        x = self._sanitize_input(x).clone()
        mean = self.norm_params["mean"][self.norm_params["mask"]]
        std = self.norm_params["std"][self.norm_params["mask"]]
        norm = (x[..., self.norm_params["mask"]] - mean) / std
        x[..., self.norm_params["mask"]] = torch.clip(norm, -self.clip, self.clip)
        return x

    def update(self, x: List | np.ndarray | torch.Tensor):
        """Update the normalizer parameters with the values in ``x``.

        Args:
            x: Batch of arrays used to update the mean and variance estimate for each entry.
        """
        # Use a batched version of Welford's algorithm for numerical stability
        x = self._sanitize_input(x)
        assert x.ndim == self.obs_dim + 1, "Input data must be a batch of arrays."
        self.count += x.shape[0]
        delta = x - self.norm_params["mean"]
        self.norm_params["mean"] += torch.sum(delta / self.count, axis=0)
        self._m2 += torch.sum(delta * (x - self.norm_params["mean"]), axis=0)
        self.norm_params["std"][:] = torch.sqrt(torch.maximum(self.eps2, self._m2 / self.count))


class ImageNormalizer(AbstractNormalizer):
    """Normalizer class for preprocessing images on both the client and the server side.

    Normalizes tensors by limiting them to a range of [-1, 1] instead of [0, 255].
    """

    def __init__(self, obs_shape: tuple[int, ...]):
        """Initialize the normalizer parameters.

        Args:
            obs_shape: Observation shape.
        """
        super().__init__()
        self.obs_shape = obs_shape
        self.obs_dim = len(obs_shape)

    def normalize(self, x: List | np.ndarray | torch.Tensor) -> torch.Tensor:
        """Normalize the input data with the current mean and variance estimate.

        Args:
            x: Input data array.

        Returns:
            The normalized data.
        """
        x = self._sanitize_input(x).clone()
        return (x / 127.5) - 1.0  # [0, 255] -> [0, 2] -> [-1, 1]

    def update(self, _: Any):
        """No-op for compatibility with the normalizer API.

        Args:
            x: Batch of arrays used to update the mean and variance estimate for each entry.
        """
