import numpy as np
from gymnasium import ObservationWrapper, Env
import einops


class ReorderChannels(ObservationWrapper):

    def __init__(self, env: Env):
        super().__init__(env)

    def observation(self, observation: np.ndarray) -> np.ndarray:
        return einops.rearrange(observation, "w h c -> c w h")
