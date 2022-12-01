import logging
from typing import Optional, List, Union
import io

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class Normalizer(nn.Module):

    def __init__(self, size_s, eps: float = 1e-2, clip: float = np.inf,
                 idx_list: Optional[List] = None):
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

    def normalize(self, x: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
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

    def update(self, x: Union[np.ndarray, torch.Tensor]):
        # Use a batched version of Welford's algorithm for numerical stability
        x = self._sanitize_input(x)
        assert x.ndim == 2
        self.count += x.shape[0]
        delta = x - self.mean
        self.mean += torch.sum(delta / self.count, axis=0)
        self._m2 += torch.sum(delta * (x - self.mean), axis=0)
        self.std[:] = torch.sqrt(torch.maximum(self.eps2, self._m2 / self.count))

    def serialize(self):
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
    def deserialize(serialization):
        idx_buff = io.BytesIO(serialization["norm.idx"])
        mean_buff = io.BytesIO(serialization["norm.mean"])
        std_buff = io.BytesIO(serialization["norm.std"])
        idx_buff.seek(0)
        mean_buff.seek(0)
        std_buff.seek(0)
        return torch.load(idx_buff), torch.load(mean_buff), torch.load(std_buff)

    def load_params(self, idx, mean, std):
        self.idx[:] = idx
        self.mean[:] = mean
        self.std[:] = std

    @staticmethod
    def _sanitize_input(x):
        if isinstance(x, torch.Tensor):
            return x.float()
        if isinstance(x, np.ndarray):
            return torch.from_numpy(x).float()
        if isinstance(x, List):
            return torch.tensor(x).float()
        raise TypeError(f"Unsupported input type {x.__class__.__name__} for normalizer")
