"""TensorDict wrapper module to convert the outputs of an environment into a TensorDict.

We use TensorDicts throughout the whole framework to serialize, store and manipulate data. This
wrapper ensures that environments are compatible with this data model.
"""
from typing import Any, Callable
import logging
import copy

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
        """Initialize the wrapper.

        Args:
            env: The environment to wrap.
            device: The device to use for the observations and actions.
        """
        super().__init__(env)
        assert isinstance(env, (VectorEnv, ExpVectorEnv)), "Only vectorized environments supported."
        self.device = device
        self.num_envs = getattr(env.unwrapped, "num_envs", 1)
        self.vectorized = getattr(env.unwrapped, "num_envs", None) is not None
        # Infer the device of the environment. If the environment action space is a numpy array,
        # we need to convert the step() action to a numpy array before passing it to the
        # environment. If the environment action space is a Tensor, we ensure that it is on the
        # correct device.
        self.env_type, self.env_device = self._env_type_and_device(env)
        self.observation_space = copy.deepcopy(env.observation_space)
        self.action_space = copy.deepcopy(env.action_space)
        self.observation_space.sample = self._patch_space(self.observation_space.sample)
        self.action_space.sample = self._patch_space(self.action_space.sample)
        self._failed_info_keys = set()  # Keep track of info keys that failed to convert

    def step(self, action: Tensor) -> TensorDict[str, Tensor]:
        """Advance one step in the environment and return the results as a TensorDict.

        Args:
            action: The action to take in the environment.

        Returns:
            A TensorDict containing the next observation, reward, termination signal, and info.
        """
        assert isinstance(action, Tensor), f"Expected action to be a Tensor, got {type(action)}."
        sample = TensorDict({"action": action}, batch_size=self.num_envs, device=self.device)
        action = self.transform_action(action)  # Convert to np if necessary or send to env_device
        next_obs, reward, terminated, truncated, info = self.env.step(action)
        sample["next_obs"] = self.transform_obs(next_obs).clone()
        sample["reward"] = torch.as_tensor(reward, dtype=torch.float64).clone()
        sample["terminated"] = torch.as_tensor(terminated, dtype=torch.bool).clone()
        sample["truncated"] = torch.as_tensor(truncated, dtype=torch.bool).clone()
        sample["info"] = self.transform_info(info).clone()
        return sample

    def reset(self,
              *,
              seed: int | None = None,
              options: dict[str, Any] | None = None) -> TensorDict[str, Tensor]:
        """Reset the environment and return the initial observation.

        Args:
            seed: Seed to use for the environment. If None, a random seed is used.
            options: Options to pass to the environment.

        Returns:
            A TensorDict containing the initial observation and info.
        """
        obs, info = self.env.reset(seed=seed, options=options)
        sample = TensorDict({}, batch_size=self.num_envs, device=self.device)
        sample["obs"] = self.transform_obs(obs).clone()
        sample["info"] = self.transform_info(info).clone()
        return sample

    def transform_obs(self,
                      obs: np.ndarray | Tensor | dict[str, np.ndarray]) -> Tensor | TensorDict:
        """Convert the observation to a Tensor or TensorDict.

        Args:
            obs: The observation to convert.

        Returns:
            The observation as a Tensor or TensorDict.
        """
        match obs:
            case np.ndarray():
                return torch.as_tensor(obs, device=self.device)
            case dict():
                return TensorDict(obs, batch_size=self.num_envs, device=self.device)
            case Tensor():
                return obs.to(self.device)
            case _:
                raise TypeError(f"Unsupported observation type {type(obs)}")

    def transform_action(self, action: Tensor) -> Tensor | np.ndarray | int:
        """Convert the action to a Tensor or np.ndarray.

        Args:
            action: The action to convert.

        Returns:
            The action as a Tensor or np.ndarray.
        """
        assert isinstance(action, Tensor), f"Expected action to be a Tensor, got {type(action)}."
        if self.env_type == "np":
            return action.cpu().numpy()
        return action.to(self.env_device)

    def transform_info(self, info: dict) -> TensorDict:
        """Convert the info to a TensorDict.

        Args:
            info: The info dictionary.

        Returns:
            The info as a TensorDict.
        """
        assert isinstance(info, dict), f"Expected dict, got {type(info)}"
        info_tf = {}
        for key, value in info.items():
            match value:
                case np.ndarray():
                    if value.dtype == np.object_:
                        info_tf[key] = self._transform_np_object(value)
                    else:
                        info_tf[key] = torch.as_tensor(value, device=self.device)
                case Tensor():
                    info_tf[key] = value.to(self.device)
                case _:
                    if key not in self._failed_info_keys:
                        self._failed_info_keys.add(key)  # Only log once per key
                        logger.warning((f"Dropping info key '{key}' with unsupported conversion "
                                        f"type {type(value)}"))
        return TensorDict(info_tf, batch_size=self.num_envs, device=self.device)

    def _transform_np_object(self, value: np.ndarray) -> dict[str, np.ndarray] | np.ndarray:
        """Convert a numpy array with dtype np.object to a list of Tensors."""
        assert isinstance(value, np.ndarray), f"Expected np.ndarray, got {type(value)}"
        assert value.dtype == object, f"Expected dtype np.object, got {value.dtype}"
        match value[0]:
            case dict():
                return {k: np.stack([d[k] for d in value]) for k in value[0].keys()}
            case np.ndarray():
                return np.stack([x for x in value])
            case _:
                raise TypeError(f"Unsupported type {type(value[0])}")

    def _patch_space(self, fn: Callable) -> Callable:

        def wrapper() -> Tensor:
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
