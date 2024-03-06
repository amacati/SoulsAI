"""Utility wrapper to make the Atari observations compatible with PyTorch."""
import numpy as np
from gymnasium import Env, ObservationWrapper


class AtariExpandImage(ObservationWrapper):
    """Expand the observation to a 3D tensor.

    The grayscale Atari environments in gymnasium return 2D observations when using the default
    wrappers. PyTorch however expects 3D tensors for convolutional layers. This wrapper expands the
    observation to the required dimensions.
    """

    def __init__(self, env: Env):
        """Initialize the wrapper.

        Args:
            env: The environment to wrap.
        """
        super().__init__(env)

    def observation(self, observation: np.ndarray) -> np.ndarray:
        """Expand the observation to a 3D tensor.

        Args:
            observation: The observation to expand.

        Returns:
            The observation as a 3D tensor.
        """
        return np.expand_dims(observation, axis=0)
