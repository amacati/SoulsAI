import numpy as np
from gymnasium import ObservationWrapper, Env
from gymnasium.wrappers.frame_stack import LazyFrames
import einops


class ReorderChannels(ObservationWrapper):

    def __init__(self, env: Env):
        super().__init__(env)

    def observation(self, observation: np.ndarray) -> np.ndarray:
        return einops.rearrange(observation, "s w h c -> (s c) w h")


class MaterializeFrames(ObservationWrapper):

    def __init__(self, env: Env):
        super().__init__(env)

    def observation(self, observation: LazyFrames) -> np.ndarray:
        return np.array(observation)
