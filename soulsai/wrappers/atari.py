import numpy as np
from gymnasium import Env, ObservationWrapper


class AtariExpandImage(ObservationWrapper):

    def __init__(self, env: Env):
        super().__init__(env)

    def observation(self, observation: np.ndarray) -> np.ndarray:
        return np.expand_dims(observation, axis=0)
