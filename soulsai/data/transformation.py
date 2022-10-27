import numpy as np
from soulsgym.core.game_state import GameState
from soulsgym.core.static import player_animations, boss_animations

from soulsai.data.one_hot_encoder import OneHotEncoder


class GameStateTransformer:

    space_coords_low = np.array([110., 540., -73.])
    space_coords_high = np.array([190., 640., -55.])
    space_coords_diff = space_coords_high - space_coords_low
    boss_move_animations = ["Walk", "Idle", "Turn", "Blend", "Falling", "Land"]

    def __init__(self):
        self.player_animation_encoder = OneHotEncoder(allow_unknown=True)
        filtered_player_animations = list(unique(map(self.filter_player_animation,
                                                     player_animations["standard"].keys())))
        self.player_animation_encoder.fit(filtered_player_animations)
        self.boss_animation_encoder = OneHotEncoder(allow_unknown=True)
        filtered_boss_animations = list(unique(map(lambda x: self.filter_boss_animation(x)[0],
                                                   boss_animations["iudex"]["all"])))
        self.boss_animation_encoder.fit(filtered_boss_animations)

        self._current_time = 0.
        self._acuumulated_time = 0.
        self._last_animation = None

    def transform(self, gamestate: GameState) -> np.ndarray:
        player_animation = self.filter_player_animation(gamestate.player_animation)
        player_animation_onehot = self.player_animation_encoder(player_animation)
        boss_animation_onehot, boss_animation_duration = self.boss_animation_transform(gamestate)
        animation_times = [gamestate.player_animation_duration, boss_animation_duration]
        return np.concatenate((self._common_transforms(gamestate), animation_times,
                               player_animation_onehot, boss_animation_onehot), dtype=np.float32)

    def stateless_transform(self, gamestate: GameState) -> np.ndarray:
        animation_times = [gamestate.player_animation_duration, gamestate.boss_animation_duration]
        player_animation = self.filter_player_animation(gamestate.player_animation)
        player_animation_onehot = self.player_animation_encoder(player_animation)
        boss_animation = self.filter_boss_animation(gamestate.boss_animation)
        boss_animation_onehot = self.boss_animation_encoder(boss_animation)
        return np.concatenate((self._common_transforms(gamestate), animation_times,
                               player_animation_onehot, boss_animation_onehot), dtype=np.float32)

    def _common_transforms(self, gamestate):
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
        self._current_time = 0.
        self._acuumulated_time = 0.
        self._last_animation = None

    def boss_animation_transform(self, gamestate: GameState):
        boss_animation, is_filtered = self.filter_boss_animation(gamestate.boss_animation)
        if not is_filtered:
            self._acuumulated_time = 0
            self._current_time = 0
            self._last_animation = boss_animation
            return self.boss_animation_encoder(boss_animation), gamestate.boss_animation_duration
        if gamestate.boss_animation != self._last_animation:
            self._last_animation = gamestate.boss_animation
            # Correct for the fact that a step takes (on average) 0.1s. If the boss animation
            # changes at the end of the step, we are missing 0.1 - new_time of animation time for
            # the accumulation.
            self._acuumulated_time = self._current_time + 0.1 - gamestate.boss_animation_duration
        boss_animation_time = gamestate.boss_animation_duration + self._acuumulated_time
        self._current_time = boss_animation_time
        return self.boss_animation_encoder(boss_animation), boss_animation_time

    @staticmethod
    def filter_player_animation(animation):
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
    def filter_boss_animation(cls, animation):
        if any([x in animation for x in cls.boss_move_animations]):
            return "Move", True
        return animation, False


def unique(seq):
    seen = set()
    return [x for x in seq if not (x in seen or seen.add(x))]


def wrap2pi(x) -> float:
    return ((x + np.pi) % (2 * np.pi)) - np.pi


def rad2vec(x):
    return np.array([np.sin(x), np.cos(x)])
