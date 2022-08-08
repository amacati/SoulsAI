import numpy as np
from soulsgym.core.game_state import GameState
from soulsgym.core.static import player_animations, boss_animations

from soulsai.core.one_hot_encoder import OneHotEncoder


def unique(seq):
    seen = set()
    seen_add = seen.add
    return [x for x in seq if not (x in seen or seen_add(x))]


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


def filter_boss_animation(animation):
    if "Attack" in animation:
        return animation
    if any([x in animation for x in ["Walk", "Idle", "Turn", "Blend", "Falling", "Land"]]):
        return "Move"
    return animation


space_coords_low = np.array([110., 540., -73.])
space_coords_high = np.array([190., 640., -55.])
space_coords_diff = space_coords_high - space_coords_low

player_animation_encoder = OneHotEncoder(allow_unknown=True)
p_animations = list(unique(map(filter_player_animation, player_animations["standard"].keys())))
player_animation_encoder.fit(p_animations)
boss_animation_encoder = OneHotEncoder(allow_unknown=True)
b_animations = list(unique(map(filter_boss_animation, boss_animations["iudex"]["all"])))
boss_animation_encoder.fit(b_animations)


def fill_buffer(buffer, env, samples, load=False, save=False, path=None):
    try:
        if load:
            if path.exists():
                buffer.load(path)
                if len(buffer) == samples:
                    return
                print("Buffer size mismatch, filling again")
                buffer.clear()
            else:
                print("Buffer save not found, filling again")
        while len(buffer) < samples:
            done = False
            state = env.reset()
            while not done:
                action = env.action_space.sample()
                next_state, reward, done, _ = env.step(action)
                buffer.append((state, action, reward, next_state, done))
                state = next_state
        if save:
            buffer.save(path)
    except:  # noqa: E722
        env.close()


def wrap2pi(x) -> float:
    return ((x + np.pi) % (2 * np.pi)) - np.pi


def _rot2sincos(x):
    return np.array([np.sin(x), np.cos(x)])


def gamestate2np(gamestate: GameState) -> np.ndarray:
    player_hp = gamestate.player_hp / gamestate.player_max_hp
    player_sp = gamestate.player_sp / gamestate.player_max_sp
    boss_hp = gamestate.boss_hp / gamestate.boss_max_hp
    player_pos = (gamestate.player_pose[:3] - space_coords_low) / space_coords_diff
    player_rot = _rot2sincos(gamestate.player_pose[3])
    boss_pos = (gamestate.boss_pose[:3] - space_coords_low) / space_coords_diff
    boss_rot = _rot2sincos(gamestate.boss_pose[3])
    camera_angle = np.arctan2(gamestate.camera_pose[3], gamestate.camera_pose[4])
    camera_rot = _rot2sincos(camera_angle)
    boss_distance = np.linalg.norm(gamestate.boss_pose[:2] - gamestate.player_pose[:2]) / 50  # noqa: E501 Normalization guess
    # cam_rot_rel = _rot2sincos(wrap2pi(gamestate.boss_pose[3] - camera_angle))
    player_animation = player_animation_encoder(filter_player_animation(gamestate.player_animation))
    player_animation = player_animation
    player_animation_duration = gamestate.player_animation_duration
    boss_animation = boss_animation_encoder(filter_boss_animation(gamestate.boss_animation))
    boss_animation = boss_animation
    boss_animation_duration = gamestate.boss_animation_duration
    combo_counter = gamestate.combo_length
    return np.concatenate(([player_hp, player_sp, boss_hp, boss_distance, combo_counter,
                            player_animation_duration, boss_animation_duration],
                            player_pos, player_rot, boss_pos, boss_rot, camera_rot, 
                            player_animation, boss_animation), dtype=np.float32)
