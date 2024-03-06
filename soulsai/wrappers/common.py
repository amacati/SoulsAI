"""Common observation wrappers for gymnasium environments."""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from gymnasium import ObservationWrapper, Env
from gymnasium.spaces import Box
import einops

if TYPE_CHECKING:
    from gymnasium.wrappers.frame_stack import LazyFrames


class ReorderChannels(ObservationWrapper):
    """Wrapper that reorders the channels of the observation.

    The reorder pattern is a string that describes the new order of the channels. It uses the einops
    `rearange` syntax. This is useful to reorder the color channels of multiple stacked frames into
    a single merged dimension.
    """

    def __init__(self, env: Env, reorder_pattern: str = "s w h c -> (s c) w h"):
        """Initialize the wrapper.

        Args:
            env: The environment to wrap.
            reorder_pattern: The pattern to reorder the channels. Assumes einops' `rearange` syntax.
        """
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
        """Reorder the channels of the observation.

        Args:
            observation: The observation to reorder.

        Returns:
            The observation with reordered channels.
        """
        return einops.rearrange(observation, self.reorder_pattern)


class MaterializeFrames(ObservationWrapper):
    """Wrapper that materializes lazy frames.

    Frame stacking wrappers in gymnasium use a custom class LazyFrames to store the observations to
    save memory. However, lazy frames are not compatible with some libraries such as einops.
    Furthermore, soulsai must materialize all observations as tensors for serialization. This
    wrapper undoes the lazy frame conversion.
    """

    def __init__(self, env: Env):
        """Initialize the wrapper.

        Args:
            env: The environment to wrap.
        """
        super().__init__(env)

    def observation(self, observation: LazyFrames) -> np.ndarray:
        """Materialize the lazy frames.

        Args:
            observation: The lazy frames to materialize.

        Returns:
            The materialized frames as a numpy array.
        """
        return np.array(observation)


class CenterCropFrames(ObservationWrapper):
    """Center crop the observation.

    Center cropping means that the same amount of pixels is removed from the left and right as well
    as from the top and bottom of the observation. The amount of pixels removed is calculated as the
    difference between the input and output shape divided by two.
    """

    def __init__(self, env: Env, output_shape: tuple[int, ...]):
        """Initialize the wrapper.

        Args:
            env: The environment to wrap.
            output_shape: The shape of the output observation.
        """
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
        """Center crop the observation.

        Args:
            observation: The observation to crop.

        Returns:
            The observation frames with removed edges.
        """
        assert observation.shape == self.input_shape
        return observation[self.slices]
