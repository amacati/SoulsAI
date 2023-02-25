"""The normalizer module implements a zero mean, unit variance normalizer.

The normalization is ensured to be numerically stable by setting a lower bound on the possible
standard deviation of values.
"""
import logging
from typing import List, Tuple
import io

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class Normalizer(nn.Module):
    """Normalizer class for preprocessing on both the client and the server side.

    Normalizes tensors to zero mean, unit variance by updating its statistics over previously seen
    values. Also includes functions to serialize the necessary parameters (e.g. server side),
    deserialize parameters (e.g. client side), and load them into the normalizer to complete an
    update.
    """

    def __init__(self, size_s: int, eps: float = 1e-2, clip: float = np.inf,
                 idx_list: List | None = None):
        """Initialize the normalizer parameters.

        Args:
            size_s: State dimension.
            eps: Minimum denominator for normalization. Enforces stability in case of low variances.
            clip: Normalization clipping value. Restricts normalized values to the interval of
                [-clip, clip].
            idx_list: List of indices to include in the normalization. If not provided, all states
                are normalized by default.
        """
        super().__init__()
        self.size = size_s
        self.clip = clip
        self.idx = nn.Parameter(torch.tensor(idx_list or range(0, size_s), dtype=torch.int64),
                                requires_grad=False)
        self.eps2 = torch.ones(size_s, dtype=torch.float32) * eps**2
        self.count = nn.Parameter(torch.tensor(0, dtype=torch.int64), requires_grad=False)
        self.mean = nn.Parameter(torch.zeros(size_s, dtype=torch.float32), requires_grad=False)
        self._m2 = nn.Parameter(torch.zeros(size_s, dtype=torch.float32), requires_grad=False)
        self.std = nn.Parameter(torch.ones(size_s, dtype=torch.float32), requires_grad=False)

    def normalize(self, x: List | np.ndarray | torch.Tensor) -> torch.Tensor:
        """Normalize the input data with the current mean and variance estimate.

        Args:
            x: Input data array.
        Returns:
            The normalized data.
        """
        x = self._sanitize_input(x).clone()
        norm = (x[..., self.idx] - self.mean[self.idx]) / self.std[self.idx]
        x[..., self.idx] = torch.clip(norm, -self.clip, self.clip)
        return x

    def update(self, x: List | np.ndarray | torch.Tensor):
        """Update the normalizer parameters with the values in ``x``.

        Args:
            x: Batch of arrays used to update the mean and variance estimate for each entry.
        """
        # Use a batched version of Welford's algorithm for numerical stability
        x = self._sanitize_input(x)
        assert x.ndim == 2
        self.count += x.shape[0]
        delta = x - self.mean
        self.mean += torch.sum(delta / self.count, axis=0)
        self._m2 += torch.sum(delta * (x - self.mean), axis=0)
        self.std[:] = torch.sqrt(torch.maximum(self.eps2, self._m2 / self.count))

    def serialize(self) -> dict:
        """Serialize the normalizer by dumping the parameter tensors as bytes into a dictionary.

        Returns:
            The dictionary containing the saved tensors.
        """
        idx_buff, mean_buff, std_buff = io.BytesIO(), io.BytesIO(), io.BytesIO()
        torch.save(self.idx, idx_buff)
        idx_buff.seek(0)
        torch.save(self.mean, mean_buff)
        mean_buff.seek(0)
        torch.save(self.std, std_buff)
        std_buff.seek(0)
        return {"norm.idx": idx_buff.read(),
                "norm.mean": mean_buff.read(),
                "norm.std": std_buff.read()}

    @staticmethod
    def deserialize(serialization: dict) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Deserialize the norm parameter buffers in the state dict.

        The dictionary is assumed to contain the ``norm.idx``, ``norm.mean`` and ``norm.std`` keys.

        Args:
            serialization: Dictionary containing the byte objects of torch tensors at predefined
                keys.
        """
        idx_buff = io.BytesIO(serialization["norm.idx"])
        mean_buff = io.BytesIO(serialization["norm.mean"])
        std_buff = io.BytesIO(serialization["norm.std"])
        idx_buff.seek(0)
        mean_buff.seek(0)
        std_buff.seek(0)
        return torch.load(idx_buff), torch.load(mean_buff), torch.load(std_buff)

    def load_params(self, idx: torch.Tensor, mean: torch.Tensor, std: torch.Tensor):
        """Load the parameter tensors into the normalizer.

        Warning:
            Parameter loading is intended to update the client normalizers and only updates the
            parameters required for normalizing. Does `NOT` update the internal ``_m2`` and
            ``count`` values.

        Args:
            idx: The index parameters.
            mean: The mean parameters.
            std: The standard deviation parameters.
        """
        self.idx[:] = idx
        self.mean[:] = mean
        self.std[:] = std

    @staticmethod
    def _sanitize_input(x: List | np.ndarray | torch.Tensor) -> torch.Tensor:
        if isinstance(x, torch.Tensor):
            return x.float()
        if isinstance(x, np.ndarray):
            return torch.from_numpy(x).float()
        if isinstance(x, List):
            return torch.tensor(x).float()
        raise TypeError(f"Unsupported input type {x.__class__.__name__} for normalizer")
