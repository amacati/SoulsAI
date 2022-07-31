import logging
from typing import Optional, Tuple, Union
import json
import multiprocessing as mp
import ctypes

import numpy as np
import torch

logger = logging.getLogger(__name__)


class Normalizer:
    
    def __init__(self, size_s, eps: float = 1e-2, clip: float = np.inf,
                 idx_range: Optional[Tuple] = None):
        self.idx = list(idx_range or (0, size_s))
        self.size = size_s
        self.eps2 = np.ones(size_s, dtype=np.float32) * eps**2
        self.clip = clip
        self.sum = np.zeros(size_s, dtype=np.float32)
        self.sum_sq = np.zeros(size_s, dtype=np.float32)
        self.count = 1
        self.mean = np.zeros(size_s, dtype=np.float32)
        self.std = np.ones(size_s, dtype=np.float32)
        self._do_normalize = 0  # Disable normalizer if only unit transform is performed
        self.shared_memory = False

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Alias for `self.normalize`."""
        return self.normalize(x)

    def normalize(self, x: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        """Normalize the input data with the current mean and variance estimate.

        Args:
            x: Input data array.
        Returns:
            The normalized data.
        """
        if not self.do_normalize:
            return x
        assert isinstance(x, np.ndarray) or isinstance(x, torch.Tensor)
        is_tensor = isinstance(x, torch.Tensor)
        x = x.copy() if isinstance(x, np.ndarray) else x.numpy().copy()
        if self.shared_memory:  # Convert shared memory arrays to np arrays first
            mean = np.frombuffer(self.mean.get_obj())[self.idx[0]:self.idx[1]]  # Avoid data copy
            std = np.frombuffer(self.std.get_obj())[self.idx[0]:self.idx[1]]
        else:
            mean, std = self.mean[self.idx[0]:self.idx[1]], self.std[self.idx[0]:self.idx[1]]
        norm = (x[..., self.idx[0]:self.idx[1]] - mean) / std
        x[..., self.idx[0]:self.idx[1]] = np.clip(norm, -self.clip, self.clip)
        return torch.as_tensor(x) if is_tensor else x

    def update(self, x: Union[np.ndarray, torch.Tensor]):
        assert isinstance(x, np.ndarray) or isinstance(x, torch.Tensor)
        if isinstance(x, torch.Tensor):
            x = x.numpy()
        assert x.ndim == 2
        self.do_normalize = True
        self.sum += np.sum(x, axis=0, dtype=np.float32)
        self.sum_sq += np.sum(x**2, axis=0, dtype=np.float32)
        self.count += x.shape[0]
        self.mean = self.sum / self.count
        self.std = (self.sum_sq / self.count - (self.sum / self.count)**2)
        np.maximum(self.eps2, self.std, out=self.std)  # Numeric stability
        np.sqrt(self.std, out=self.std)

    def serialize(self):
        return {"norm.idx": np.array(self.idx).tobytes(),
                "norm.mean": self.mean.tobytes(),
                "norm.std": self.std.tobytes(),
                "norm.do_normalize": self.do_normalize}

    def deserialize(self, serialization):
        self.do_normalize = serialization["norm.do_normalize"]
        if self.do_normalize:
            self.idx[:] = np.frombuffer(serialization["norm.idx"], dtype=np.int64)
            self.mean[:] = np.frombuffer(serialization["norm.mean"])
            self.std[:] = np.frombuffer(serialization["norm.std"])

    def state_dict(self):
        return {"shared_memory": self.shared_memory, "do_normalize": self.do_normalize,
                "idx": self.idx, "mean": self.mean, "std": self.std}

    def load_state_dict(self, state_dict):
        if not self.shared_memory and state_dict["shared_memory"]:
            self.share_memory()
        self.do_normalize = state_dict["do_normalize"]
        self.idx[:] = state_dict["idx"]
        self.mean[:] = state_dict["mean"]
        self.std[:] = state_dict["std"]

    def share_memory(self):
        # Only mean and std have to be shared, the rest is only used for updating, which is not used
        # on the clients
        assert not self.shared_memory
        idx = mp.Array(ctypes.c_int64, 2)
        idx[:] = self.idx
        self.idx = idx
        mean = mp.Array(ctypes.c_double, len(self.mean))
        mean[:] = self.mean
        self.mean = mean
        std = mp.Array(ctypes.c_double, len(self.std))
        std[:] = self.std
        self.std = std
        do_normalize = self.do_normalize
        self._do_normalize = mp.Value("i", do_normalize)
        self.shared_memory = True

    def load(self, path):
        with open(path, "r") as f:
            save_dict = json.load(f)
        for key, value in save_dict.items():
            setattr(self, key[4:], np.array(value))

    def save(self, path):
        save_dict = {"norm.idx": self.idx, "norm.do_normalize": self.do_normalize}
        save_dict["norm.mean"] = list(np.float64(self.mean))
        save_dict["norm.std"] = list(np.float64(self.std))
        with open(path, "w") as f:
            json.dump(save_dict, f)

    @property
    def do_normalize(self):
        if self.shared_memory:
            return self._do_normalize.value
        return self._do_normalize

    @do_normalize.setter
    def do_normalize(self, value):
        if self.shared_memory:
            self._do_normalize.value = int(value)
        else:
            self._do_normalize = int(value)