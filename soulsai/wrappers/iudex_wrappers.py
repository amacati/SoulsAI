from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
from gymnasium import ObservationWrapper
from gymnasium.spaces import Box
from soulsgym.games.game import StaticGameData

from soulsai.data.one_hot_encoder import OneHotEncoder


class IudexObservationWrapper(ObservationWrapper):

    space_coords_low = np.array([110., 540., -73.])
    space_coords_high = np.array([190., 640., -55.])
    space_coords_diff = space_coords_high - space_coords_low

    def __init__(self, env):
        super().__init__(env)
        self.game_id = "DarkSoulsIII"
        self.boss_id = "iudex"
        low = np.zeros(74, dtype=np.float32)
        high = np.ones(74, dtype=np.float32)
        # Boss distance, player animation duration, boss animation duration are possibly unbounded
        high[[3, 16, 17]] = np.inf
        self.observation_space = Box(low, high, dtype=np.float32)
        self.game_data = StaticGameData("DarkSoulsIII")
        # Initialize player one-hot encoder
        self.player_animation_encoder = OneHotEncoder(allow_unknown=True)
        p_animations = [a["ID"] for a in self.game_data.player_animations.values()]
        filtered_player_animations = unique(map(self.filter_player_animation, p_animations))
        self.player_animation_encoder.fit(filtered_player_animations)
        # Initialize boss one-hot encoder
        self.boss_animation_encoder = OneHotEncoder(allow_unknown=True)
        iudex_animations = self.game_data.boss_animations[self.boss_id]["all"]
        boss_animations = [a["ID"] for a in iudex_animations.values()]
        filtered_boss_animations = unique(
            map(lambda x: self.filter_boss_animation(x)[0], boss_animations))
        self.boss_animation_encoder.fit(filtered_boss_animations)
        # Initialize stateful attributes
        self._current_time = 0.
        self._acuumulated_time = 0.
        self._last_animation = None

    def observation(self, obs: Dict) -> np.ndarray:
        """Transform a game observation with a stateful conversion.

        Warning:
            This function is assumed to be called with successive ``SoulsGym`` observations. If the
            next observation is not part of the same trajectory,
            :meth:`.IudexObservationWrapper.reset` has to be called.

        Args:
            obs: The input observation.

        Returns:
            A transformed observation as a numerical array.
        """
        # The final observation has the following entries:
        # 0-3: player_hp, player_sp, boss_hp, boss_distance
        # 4-15: player_pos, player_rot, boss_pos, boss_rot, camera_rot
        # 16-17: player_animation_duration, boss_animation_duration
        # 17-48: player_animation_onehot
        # 49-73: boss_animation_onehot
        obs = self._unpack_obs(obs)
        player_animation = self.filter_player_animation(obs["player_animation"])
        player_animation_onehot = self.player_animation_encoder(player_animation)
        boss_animation_onehot, boss_animation_duration = self.boss_animation_transform(obs)
        animation_times = [obs["player_animation_duration"], boss_animation_duration]
        return np.concatenate((self._common_transforms(obs), animation_times,
                               player_animation_onehot, boss_animation_onehot),
                              dtype=np.float32)

    def _common_transforms(self, obs: Dict) -> np.ndarray:
        player_hp = obs["player_hp"] / obs["player_max_hp"]
        player_sp = obs["player_sp"] / obs["player_max_sp"]
        boss_hp = obs["boss_hp"] / obs["boss_max_hp"]
        player_pos = (obs["player_pose"][:3] - self.space_coords_low) / self.space_coords_diff
        player_rot = rad2vec(obs["player_pose"][3])
        boss_pos = (obs["boss_pose"][:3] - self.space_coords_low) / self.space_coords_diff
        boss_rot = rad2vec(obs["boss_pose"][3])
        camera_angle = np.arctan2(obs["camera_pose"][3], obs["camera_pose"][4])
        camera_rot = rad2vec(camera_angle)
        # 50 is a normalization guess
        boss_distance = np.linalg.norm(obs["boss_pose"][:2] - obs["player_pose"][:2]) / 50
        args = (
            [player_hp, player_sp, boss_hp, boss_distance],
            player_pos,
            player_rot,
            boss_pos,
            boss_rot,
            camera_rot,
        )
        return np.concatenate(args, dtype=np.float32)

    def reset(self,
              *,
              seed: int | None = None,
              options: dict[str, Any] | None = None) -> tuple[Dict, dict[str, Any]]:
        """Reset the stateful attributed of the transformer.

        Modifies the :attr:`env` after calling :meth:`reset`, returning a modified observation using
        :meth:`self.observation`.
        """
        self._current_time = 0.
        self._acuumulated_time = 0.
        self._last_animation = None
        obs, info = self.env.reset(seed=seed, options=options)
        return self.observation(obs), info

    def boss_animation_transform(self, obs: Dict) -> Tuple[np.ndarray, float]:
        """Transform the observation's boss animation into a one-hot encoding and a duration.

        Since we are binning the boss animations, we have to sum the durations of binned animations.
        This requires the transformer to be stateful to keep track of previous animations.

        Note:
            To correctly transform animations after an episode has finished, users have to call
            :meth:`.GameStateTransformer.reset` in between.

        Args:
            obs: The input observation.

        Returns:
            A tuple of the current animation as one-hot encoding and the animation duration.
        """
        boss_animation, is_filtered = self.filter_boss_animation(obs["boss_animation"])
        if not is_filtered:
            self._acuumulated_time = 0.
            self._current_time = 0.
            self._last_animation = boss_animation
            return self.boss_animation_encoder(boss_animation), obs["boss_animation_duration"]
        if obs["boss_animation"] != self._last_animation:
            self._last_animation = obs["boss_animation"]
            # The animation has changed. obs["boss_animation_duration"] now contains the duration
            # of the new animation. We have to calculate the final duration of the previous
            # animation by adding the time from the step at t-1 until the animation first changed to
            # the accumulated time.
            remaining_duration = self.env.step_size - obs["boss_animation_duration"]
            self._acuumulated_time = self._current_time + remaining_duration
        boss_animation_time = obs["boss_animation_duration"] + self._acuumulated_time
        self._current_time = boss_animation_time
        return self.boss_animation_encoder(boss_animation), boss_animation_time

    @staticmethod
    def filter_player_animation(animation: int) -> int:
        """Bin common player animations.

        Player animations that essentially constitute the same state are binned into a single
        category to reduce the state space. The new labels are in the range of 1xx to avoid
        collisions with other animation labels.

        Args:
            animation: Player animation ID.

        Returns:
            The binned player animation.
        """
        if animation in [0, 1, 2, 3, 4]:  # <Add-x> animations
            return 100
        if animation in [17, 18, 19, 23, 24, 25, 26]:  # <Idle, Move, None, Run-x>
            return 101
        if animation in [27, 28]:  # <Quick-x>
            return 102
        if animation in [39, 40]:  # <LandLow, Land>
            return 103
        if animation in [41, 42]:  # <LandFaceDown, LandFaceUp>
            return 104
        if animation in [43, 44, 45, 46, 47, 48, 49]:  # <Fall-x>
            return 105
        return animation

    def filter_boss_animation(self, animation: int) -> Tuple[int, bool]:
        """Bin boss movement animations into a single animation.

        Boss animations that essentially constitute the same state are binned into a single category
        to reduce the state space. The new labels are in the range of 1xx to avoid collisions with
        other animation labels.

        Args:
            animation: Boss animation ID.

        Returns:
            The animation name and a flag set to True if it was binned (else False).
        """
        # <WalkFront, WalkLeft, WalkRight, WalkBack, TurnRight90, TurnRight180, TurnLeft90,
        # TurnLeft180>
        if animation in [19, 20, 21, 22, 24, 25, 26, 27]:
            return 100, True
        return animation, False

    @staticmethod
    def _unpack_obs(obs: Dict) -> Dict:
        """Unpack numpy arrays of float observations.

        Args:
            obs: The initial observation.

        Returns:
            The observation with unpacked floats.
        """
        scalars = [
            "player_hp", "player_sp", "boss_hp", "boss_animation_duration",
            "player_animation_duration"
        ]
        for key in scalars:
            if isinstance(obs[key], np.ndarray):
                obs[key] = obs[key].item()
        arrays = ["player_pose", "boss_pose", "camera_pose"]
        for key in arrays:
            if not isinstance(obs[key], np.ndarray):
                assert isinstance(obs[key], list)
                obs[key] = np.array(obs[key])
        return obs


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
