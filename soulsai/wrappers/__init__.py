from gymnasium.wrappers.atari_preprocessing import AtariPreprocessing

from soulsai.wrappers.atari import AtariExpandImage
# Docker containers for testing don't install soulsgym. Wrappers that depend on it are only imported
# if soulsgym is installed.
try:
    from soulsai.wrappers.iudex_wrappers import IudexObservationWrapper
    soulsgym_installed = True
except ModuleNotFoundError:
    soulsgym_installed = False

__all__ = [AtariPreprocessing, AtariExpandImage]
if soulsgym_installed:
    __all__.append(IudexObservationWrapper)
