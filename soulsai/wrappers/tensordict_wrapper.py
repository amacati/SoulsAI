"""TensorDict wrapper module to convert the outputs of an environment into a TensorDict.

We use TensorDicts throughout the whole framework to serialize, store and manipulate data. This
wrapper ensures that environments are compatible with this data model.
"""
from __future__ import annotations

from typing import Any, Callable, TYPE_CHECKING
import logging
import copy

import torch
from torch import Tensor
from tensordict import TensorDict, set_lazy_legacy
import numpy as np
from gymnasium import Wrapper, Env

if TYPE_CHECKING:
    from gymnasium.vector import VectorEnv
    from gymnasium.experimental.vector import VectorEnv as ExpVectorEnv

logger = logging.getLogger(__name__)


class TensorDictWrapper(Wrapper):
    """A wrapper that converts the outputs into a tensordict.

    If the environment expects numpy arrays, actions are converted to numpy arrays before being
    passed to the environment. If the environment expects Tensors, the actions are sent to the
    device of the environment. If both the environment and the training are on the same device, this
    wrapper is a no-op. Observations are always converted to Tensors on the training device.
    """

    def __init__(self,
                 env: VectorEnv | ExpVectorEnv | Env,
                 device: torch.device = torch.device("cpu")):
        """Initialize the wrapper.

        Args:
            env: The environment to wrap.
            device: The device to use for the observations and actions.
        """
        super().__init__(env)
        set_lazy_legacy(False)
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
        # Convert action to np if necessary or send to env_device. If the environment is not
        # vectorized, we also extract the first element of the action
        action = self.transform_action(action)
        next_obs, reward, terminated, truncated, info = self.env.step(action)
        sample["next_obs"] = self.transform_obs(next_obs).clone()
        sample["reward"] = self.transform_reward(reward).clone()
        sample["terminated"] = self.transform_done(terminated).clone()
        sample["truncated"] = self.transform_done(truncated).clone()
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

        If the environment is not vectorized, we add a batch dimension to the observation.

        Args:
            obs: The observation to convert.

        Returns:
            The observation as a Tensor or TensorDict.
        """
        match obs:
            case np.ndarray():
                obs = torch.as_tensor(obs, device=self.device)
            case dict():
                obs = TensorDict(obs, batch_size=self.num_envs, device=self.device)
            case Tensor():
                obs = obs.to(self.device)
            case _:
                raise TypeError(f"Unsupported observation type {type(obs)}")
        if self.vectorized:
            return obs
        return obs.unsqueeze(0)  # Add batch dimension

    def transform_action(self, action: Tensor) -> Tensor | np.ndarray | int:
        """Convert the action to a Tensor or np.ndarray.

        If the environment is not vectorized, we remove the batch dimension from the action.

        Args:
            action: The action to convert.

        Returns:
            The action as a Tensor or np.ndarray.
        """
        assert isinstance(action, Tensor), f"Expected action to be a Tensor, got {type(action)}."
        action = action.cpu().numpy() if self.env_type == "np" else action.to(self.env_device)
        if self.vectorized:
            return action
        return action[0]

    def transform_reward(self, reward: float | np.ndarray) -> Tensor:
        """Convert the reward to a Tensor.

        If the environment is not vectorized, we add a batch dimension to the reward.

        Args:
            reward: The reward to convert.

        Returns:
            The reward as a Tensor.
        """
        match reward:
            case np.ndarray():
                assert self.vectorized, "Array rewards only supported in vectorized envs"
                return torch.as_tensor(reward, dtype=torch.float64, device=self.device)
            case Tensor():
                assert self.vectorized, "Array rewards only supported in vectorized envs"
                return reward.to(torch.float64).to(self.device)
            case int() | float():
                assert not self.vectorized, "Scalar rewards not supported in vectorized envs"
                return torch.tensor([reward], dtype=torch.float64, device=self.device)
            case _:
                raise TypeError(f"Unsupported reward type {type(reward)}")

    def transform_done(self, done: bool | np.ndarray | Tensor) -> Tensor:
        """Convert terminated and truncated to a Tensor.

        If the environment is not vectorized, we add a batch dimension to the done signal.

        Args:
            done: The termination signal to convert.

        Returns:
            The termination signal as a Tensor.
        """
        match done:
            case np.ndarray():
                assert self.vectorized, "Array done signals only supported in vectorized envs"
                return torch.as_tensor(done, dtype=torch.bool, device=self.device)
            case Tensor():
                assert self.vectorized, "Array done signals only supported in vectorized envs"
                return done.to(torch.bool).to(self.device)
            case bool():
                assert not self.vectorized, "Scalar done signals not supported in vectorized envs"
                return torch.tensor([done], dtype=torch.bool, device=self.device)
            case _:
                raise TypeError(f"Unsupported done signal type {type(done)}")

    def transform_info(self, info: dict) -> TensorDict:
        """Convert the info to a TensorDict.

        If the environment is not vectorized, we add a batch dimension to the info.

        Args:
            info: The info dictionary.

        Returns:
            The info as a TensorDict.
        """
        assert isinstance(info, dict), f"Expected dict, got {type(info)}"
        info_tf = {}
        # We cannot add scalar tensors without dimension to a TensorDict. However, we need to
        # unsqueeze all tensors if the environment is not vectorized. If we add the scalars before
        # unsqueezing, we effectively add two dimensions instead of one. Therefore, we first add the
        # arrays, then unsqueeze the whole tensor dict, and finally add the (manually unsqueezed)
        # scalars.
        for key, value in info.items():
            match value:
                case np.ndarray():
                    if value.dtype == np.object_:
                        info_tf[key] = self._transform_np_object(value)
                    else:
                        info_tf[key] = torch.as_tensor(value, device=self.device)
                case Tensor():
                    info_tf[key] = value.to(self.device)
                case int() | float() | str() | bool():
                    continue  # We wait until we can add scalar values to TensorDict
                case _:
                    if key not in self._failed_info_keys:
                        self._failed_info_keys.add(key)  # Only log once per key
                        logger.warning((f"Dropping info key '{key}' with unsupported conversion "
                                        f"type {type(value)}"))
        info = TensorDict(info_tf, batch_size=self.num_envs, device=self.device)
        if self.vectorized:
            return info
        info = info.unsqueeze(0)  # Add batch dimension
        for key, value in info.items():
            if isinstance(value, (int, float, str, bool)):
                info[key] = torch.tensor([value], device=self.device)  # Add batch dimension
        return info

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
