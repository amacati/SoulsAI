from gymnasium.wrappers import ResizeObservation, FrameStack, AtariPreprocessing

from soulsai.wrappers.common import ReorderChannels
from soulsai.wrappers.atari import AtariExpandImage
# Docker containers for testing don't install soulsgym. Wrappers that depend on it are only imported
# if soulsgym is installed.
try:
    from soulsai.wrappers.iudex_wrappers import IudexObservationWrapper
    soulsgym_installed = True
except ModuleNotFoundError:
    soulsgym_installed = False

__all__ = [ReorderChannels, ResizeObservation, FrameStack, AtariPreprocessing, AtariExpandImage]
if soulsgym_installed:
    __all__.append(IudexObservationWrapper)
