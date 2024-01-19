from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from gymnasium import ObservationWrapper, Env
from gymnasium.spaces import Box
import einops

if TYPE_CHECKING:
    from gymnasium.wrappers.frame_stack import LazyFrames


class ReorderChannels(ObservationWrapper):

    def __init__(self, env: Env, reorder_pattern: str = "s w h c -> (s c) w h"):
        super().__init__(env)
        assert isinstance(self.observation_space, Box)
        self.reorder_pattern = reorder_pattern
        # Calculate new observation space
        low, high = self.observation_space.low, self.observation_space.high
        if isinstance(low, np.ndarray):
            low = einops.rearrange(low, reorder_pattern)
        if isinstance(high, np.ndarray):
            high = einops.rearrange(high, reorder_pattern)
        shape = einops.rearrange(self.observation_space.sample(), reorder_pattern).shape
        dtype = self.observation_space.dtype
        self.observation_space = Box(low=low, high=high, shape=shape, dtype=dtype)

    def observation(self, observation: np.ndarray) -> np.ndarray:
        return einops.rearrange(observation, self.reorder_pattern)


class MaterializeFrames(ObservationWrapper):

    def __init__(self, env: Env):
        super().__init__(env)

    def observation(self, observation: LazyFrames) -> np.ndarray:
        return np.array(observation)


class CenterCropFrames(ObservationWrapper):

    def __init__(self, env: Env, output_shape: tuple[int, ...]):
        super().__init__(env)
        assert isinstance(self.observation_space, Box)
        input_shape = env.observation_space.sample().shape
        assert len(input_shape) == len(output_shape)
        assert all(i >= o for i, o in zip(input_shape, output_shape))
        self.input_shape = input_shape
        left_crop = [(i - o) // 2 for i, o in zip(input_shape, output_shape)]
        self.slices = tuple(slice(i, i + j) for i, j in zip(left_crop, output_shape))
        # Calculate new observation space
        low = self.observation_space.low[self.slices]
        high = self.observation_space.high[self.slices]
        obs_shape = self.observation_space.shape
        shape = [
            s.indices(obs_shape[i])[1] - s.indices(obs_shape[i])[0]
            for i, s in enumerate(self.slices)
        ]
        dtype = self.observation_space.dtype
        self.observation_space = Box(low=low, high=high, shape=shape, dtype=dtype)

    def observation(self, observation: np.ndarray) -> np.ndarray:
        assert observation.shape == self.input_shape
        return observation[self.slices]
