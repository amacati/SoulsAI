"""The transformation module allows the transformation of ``GameState`` s from ``SoulsGym`` envs."""
from __future__ import annotations

from typing import List, Iterable, Tuple

import numpy as np
from soulsgym.core.game_state import GameState
from soulsgym.core.static import player_animations, boss_animations

from soulsai.data.one_hot_encoder import OneHotEncoder


class GameStateTransformer:
    """Transform ``GameState`` s into a numerical representation.

    The transformer allows the consistent binning of animations and encodes pose data into a more
    suitable representation.
    """

    SOULSGYM_STEP_TIME = 0.1
    space_coords_low = np.array([110., 540., -73.])
    space_coords_high = np.array([190., 640., -55.])
    space_coords_diff = space_coords_high - space_coords_low
    boss_move_animations = ["Walk", "Idle", "Turn", "Blend", "Falling", "Land"]

    def __init__(self):
        """Initialize the one-hot encoders and set up attributes for the stateful transformation."""
        self.player_animation_encoder = OneHotEncoder(allow_unknown=True)
        filtered_player_animations = unique(map(self.filter_player_animation,
                                                player_animations["standard"].keys()))
        self.player_animation_encoder.fit(filtered_player_animations)
        self.boss_animation_encoder = OneHotEncoder(allow_unknown=True)
        filtered_boss_animations = unique(map(lambda x: self.filter_boss_animation(x)[0],
                                              boss_animations["iudex"]["all"]))
        self.boss_animation_encoder.fit(filtered_boss_animations)

        self._current_time = 0.
        self._acuumulated_time = 0.
        self._last_animation = None

    def transform(self, gamestate: GameState) -> np.ndarray:
        """Transform a gamestate with a stateful conversion.

        Warning:
            This function is assumed to be called with successive ``gamestate`` s. If the next
            ``gamestate`` is not part of the same trajectory, :meth:`.GameStateTransformer.reset`
            has to be called.

        Args:
            gamestate: The input gamestate.

        Returns:
            A transformed gamestate as a numerical array.
        """
        player_animation = self.filter_player_animation(gamestate.player_animation)
        player_animation_onehot = self.player_animation_encoder(player_animation)
        boss_animation_onehot, boss_animation_duration = self.boss_animation_transform(gamestate)
        animation_times = [gamestate.player_animation_duration, boss_animation_duration]
        return np.concatenate((self._common_transforms(gamestate), animation_times,
                               player_animation_onehot, boss_animation_onehot), dtype=np.float32)

    def stateless_transform(self, gamestate: GameState) -> np.ndarray:
        """Transform a ``gamestate`` with a stateless conversion.

        Boss and player animations are filtered and binned, but not accumulated correctly.

        Args:
            gamestate: The input gamestate.

        Returns:
            A transformed gamestate as a numerical array.
        """
        animation_times = [gamestate.player_animation_duration, gamestate.boss_animation_duration]
        player_animation = self.filter_player_animation(gamestate.player_animation)
        player_animation_onehot = self.player_animation_encoder(player_animation)
        boss_animation = self.filter_boss_animation(gamestate.boss_animation)[0]
        boss_animation_onehot = self.boss_animation_encoder(boss_animation)
        return np.concatenate((self._common_transforms(gamestate), animation_times,
                               player_animation_onehot, boss_animation_onehot), dtype=np.float32)

    def _common_transforms(self, gamestate: GameState) -> np.ndarray:
        player_hp = gamestate.player_hp / gamestate.player_max_hp
        player_sp = gamestate.player_sp / gamestate.player_max_sp
        boss_hp = gamestate.boss_hp / gamestate.boss_max_hp
        player_pos = (gamestate.player_pose[:3] - self.space_coords_low) / self.space_coords_diff
        player_rot = rad2vec(gamestate.player_pose[3])
        boss_pos = (gamestate.boss_pose[:3] - self.space_coords_low) / self.space_coords_diff
        boss_rot = rad2vec(gamestate.boss_pose[3])
        camera_angle = np.arctan2(gamestate.camera_pose[3], gamestate.camera_pose[4])
        camera_rot = rad2vec(camera_angle)
        boss_distance = np.linalg.norm(gamestate.boss_pose[:2] - gamestate.player_pose[:2]) / 50  # noqa: E501 Normalization guess
        return np.concatenate(([player_hp, player_sp, boss_hp, boss_distance],
                               player_pos, player_rot, boss_pos, boss_rot, camera_rot,),
                              dtype=np.float32)

    def reset(self):
        """Reset the stateful attributed of the transformer.

        See :meth:`.GameStateTransformer.boss_animation_transform`.
        """
        self._current_time = 0.
        self._acuumulated_time = 0.
        self._last_animation = None

    def boss_animation_transform(self, gamestate: GameState) -> Tuple[np.ndarray, float]:
        """Transform the ``gamestate``'s boss animation into a one-hot encoding and a duration.

        Since we are binning the boss animations, we have to sum the durations of binned animations.
        This requires the transformer to be stateful to keep track of previous animations.

        Note:
            To correctly transform animations after an episode has finished, users have to call
            :meth:`.GameStateTransformer.reset` in between.

        Args:
            gamestate: The input gamestate.

        Returns:
            A tuple of the current animation as one-hot encoding and the animation duration.
        """
        boss_animation, is_filtered = self.filter_boss_animation(gamestate.boss_animation)
        if not is_filtered:
            self._acuumulated_time = 0.
            self._current_time = 0.
            self._last_animation = boss_animation
            return self.boss_animation_encoder(boss_animation), gamestate.boss_animation_duration
        if gamestate.boss_animation != self._last_animation:
            self._last_animation = gamestate.boss_animation
            # The animation has changed. GameState.boss_animation_duration now contains the duration
            # of the new animation. We have to calculate the final duration of the previous
            # animation by adding the time from the step at t-1 until the animation first changed to
            # the accumulated time.
            remaining_duration = self.SOULSGYM_STEP_TIME - gamestate.boss_animation_duration
            self._acuumulated_time = self._current_time + remaining_duration
        boss_animation_time = gamestate.boss_animation_duration + self._acuumulated_time
        self._current_time = boss_animation_time
        return self.boss_animation_encoder(boss_animation), boss_animation_time

    @staticmethod
    def filter_player_animation(animation: str) -> str:
        """Bin common player animations.

        Args:
            animation: Player animation.

        Returns:
            The binned player animation.
        """
        if "Add" in animation:
            return "Add"
        if "Run" in animation or animation in ["DashStop", "Idle", "Move", "None"]:
            return "Move"
        if "Quick" in animation:
            return "Quick"
        if animation in ["LandLow", "Land"]:
            return "Land"
        if animation in ["LandFaceDown", "LandFaceUp"]:
            return "LandF"
        if "Fall" in animation:
            return "Fall"
        return animation

    @classmethod
    def filter_boss_animation(cls: GameStateTransformer, animation: str) -> Tuple[str, bool]:
        """Bin boss movement animations into a single animation.

        Args:
            animation: Animation name.

        Returns:
            The animation name and a flag set to True if it was binned (else False).
        """
        if any([x in animation for x in cls.boss_move_animations]):
            return "Move", True
        return animation, False


def unique(seq: Iterable) -> List:
    """Create a list of unique elements from an iterable.

    Args:
        seq: Iterable sequence.

    Returns:
        The list of unique items.
    """
    seen = set()
    return [x for x in seq if not (x in seen or seen.add(x))]


def wrap2pi(x: float) -> float:
    """Project an angle in radians to the interval of [-pi, pi].

    Args:
        x: The angle in radians.

    Returns:
        The angle restricted to the interval of [-pi, pi].
    """
    return ((x + np.pi) % (2 * np.pi)) - np.pi


def rad2vec(x: float) -> np.ndarray:
    """Convert an angle in radians to a [sin, cos] vector.

    Args:
        x: The angle in radians.

    Returns:
        The encoded orientation as [sin, cos] vector.
    """
    return np.array([np.sin(x), np.cos(x)])
