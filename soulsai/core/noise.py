from abc import ABC, abstractmethod

import numpy as np


class Noise(ABC):

    def __init__(self):
        self.np_random = np.random.default_rng()

    @abstractmethod
    def sample(self):
        """Generate a single noise sample."""

    @abstractmethod
    def reset(self):
        """Reset the state of the noise.

        For stateless noises, this should be a no-op.
        """


class UniformDiscreteNoise(Noise):

    def __init__(self, size_n):
        super().__init__()
        assert size_n > 0
        self.size_n = size_n

    def sample(self):
        return self.np_random.integers(0, self.size_n).item()

    def reset(self):
        ...
