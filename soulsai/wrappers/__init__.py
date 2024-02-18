from gymnasium.wrappers import ResizeObservation, FrameStack, AtariPreprocessing
from gymnasium.wrappers.gray_scale_observation import GrayScaleObservation

from soulsai.wrappers.common import ReorderChannels, MaterializeFrames, CenterCropFrames
from soulsai.wrappers.atari import AtariExpandImage
from soulsai.wrappers.tensordict_wrapper import TensorDictWrapper
# Docker containers for testing don't install soulsgym. Wrappers that depend on it are only imported
# if soulsgym is installed.
try:
    from soulsai.wrappers.iudex_wrappers import IudexObservationWrapper
    soulsgym_installed = True
except ModuleNotFoundError:
    soulsgym_installed = False

__all__ = [
    "ReorderChannels", "MaterializeFrames", "ResizeObservation", "FrameStack", "AtariPreprocessing",
    "AtariExpandImage", "CenterCropFrames", "GrayScaleObservation", "TensorDictWrapper"
]
if soulsgym_installed:
    __all__.append("IudexObservationWrapper")
