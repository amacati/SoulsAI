from typing import Any, Callable
import logging

from gymnasium import Wrapper, Env
from gymnasium.vector import VectorEnv
from gymnasium.experimental.vector import VectorEnv as ExpVectorEnv
from tensordict import TensorDict

import torch
from torch import Tensor
import numpy as np

logger = logging.getLogger(__name__)


class TensorDictWrapper(Wrapper):
    """A wrapper that converts the outputs into a tensordict.

    If the environment expects numpy arrays, actions are converted to numpy arrays before being
    passed to the environment. If the environment expects Tensors, the actions are sent to the
    device of the environment. If both the environment and the training are on the same device, this
    wrapper is a no-op. Observations are always converted to Tensors on the training device.
    """

    def __init__(self, env: VectorEnv | ExpVectorEnv, device: torch.device = torch.device("cpu")):
        super().__init__(env)
        assert isinstance(env, (VectorEnv, ExpVectorEnv)), "Only vectorized environments supported."
        self.device = device
        # Infer the device of the environment. If the environment action space is a numpy array,
        # we need to convert the step() action to a numpy array before passing it to the
        # environment. If the environment action space is a Tensor, we ensure that it is on the
        # correct device
        self.env_device = torch.device("cpu")
        if isinstance(sample_action := self.env.action_space.sample(), Tensor):
            self.env_device = sample_action.device

        self.num_envs = getattr(env.unwrapped, "num_envs", 1)
        self.vectorized = getattr(env.unwrapped, "num_envs", None) is not None
        self.env_type, self.env_device = self._env_type_and_device(env)
        self.observation_space.sample = self._patch_space(self.observation_space.sample)
        self.action_space.sample = self._patch_space(self.action_space.sample)
        self._use_info = True

    def step(self, action: Tensor) -> TensorDict[str, Tensor]:
        assert isinstance(action, Tensor), f"Expected action to be a Tensor, got {type(action)}."
        action = self.transform_action(action)  # Convert to np if necessary or send to env_device
        next_obs, reward, terminated, truncated, info = self.env.step(action)
        sample = TensorDict(
            {
                "action": torch.as_tensor(action),
                "next_obs": torch.as_tensor(next_obs),
                "reward": torch.as_tensor(reward),
                "terminated": torch.as_tensor(terminated),
                "truncated": torch.as_tensor(truncated)
            },
            batch_size=self.num_envs,
            device=self.device)
        if self._use_info and info:
            try:
                sample["info"] = TensorDict(info, batch_size=self.num_envs, device=self.device)
            except TypeError:
                logger.warning("Info dict cannot be converted to TensorDict, disabling info")
                self._use_info = False
        return sample

    def reset(self,
              *,
              seed: int | None = None,
              options: dict[str, Any] | None = None) -> TensorDict[str, Tensor]:
        obs, info = self.env.reset(seed=seed, options=options)
        obs = torch.as_tensor(obs)
        if not self.vectorized:
            obs = obs.view(self.num_envs, *obs.shape)
        sample = TensorDict(
            {
                "obs": obs,
                "terminated": torch.zeros(self.num_envs, dtype=torch.bool),
                "truncated": torch.zeros(self.num_envs, dtype=torch.bool)
            },
            batch_size=self.num_envs,
            device=self.device)
        if self._use_info and info:
            try:
                sample["info"] = TensorDict(info, batch_size=self.num_envs, device=self.device)
            except TypeError:
                logger.warning("Info dict cannot be converted to TensorDict, disabling info")
                self._use_info = False
        return sample

    def transform_action(self, action: Tensor) -> Tensor | np.ndarray | int:
        if self.env_type == "np":
            return action.cpu().numpy()
        return action.to(self.env_device)

    def _patch_space(self, fn: Callable) -> Callable:

        def wrapper():
            return torch.as_tensor(fn(), device=self.device)

        return wrapper

    def _env_type_and_device(self, env: Env) -> tuple[str, torch.device]:
        """Determine the input type and device of the environment.

        Some environments expect numpy arrays as input, others expect Tensors. If Tensors are
        expected, we ensure that they are on the correct device to avoid unnecessary data transfers.
        """
        action = env.action_space.sample()
        if isinstance(action, (np.ndarray, np.generic)):
            return "np", torch.device("cpu")
        elif isinstance(action, Tensor):
            return "torch", action.device
        raise TypeError(f"Unsupported action space {type(action)}")
