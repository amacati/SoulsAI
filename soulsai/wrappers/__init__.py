"""Wrapper modules for gymnasium environments.

Some environments require additional data preprocessing steps to make them compatible with the
`soulsai` framework. This module contains several useful wrappers that can be directly specified in
the configuration file of a training run.
"""

from gymnasium.wrappers import AtariPreprocessing, FrameStack, ResizeObservation
from gymnasium.wrappers.gray_scale_observation import GrayScaleObservation

from soulsai.wrappers.atari import AtariExpandImage
from soulsai.wrappers.common import CenterCropFrames, MaterializeFrames, ReorderChannels
from soulsai.wrappers.tensordict_wrapper import TensorDictWrapper

# Docker containers for testing don't install soulsgym. Wrappers that depend on it are only imported
# if soulsgym is installed.
try:
    from soulsai.wrappers.iudex_wrappers import IudexObservationWrapper  # noqa: F401

    soulsgym_installed = True
except ModuleNotFoundError:
    soulsgym_installed = False

__all__ = [
    "ReorderChannels",
    "MaterializeFrames",
    "ResizeObservation",
    "FrameStack",
    "AtariPreprocessing",
    "AtariExpandImage",
    "CenterCropFrames",
    "GrayScaleObservation",
    "TensorDictWrapper",
]
if soulsgym_installed:
    __all__.append("IudexObservationWrapper")
