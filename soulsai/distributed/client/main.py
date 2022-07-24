import logging
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import yaml
import gym
import soulsgym  # noqa: F401
from soulsgym.core.game_state import GameState
import keyboard

from soulsai.core.utils import gamestate2np
from soulsai.distributed.client.connector import Connector

logger = logging.getLogger(__name__)


def load_config():
    root_dir = Path(__file__).parent
    with open(root_dir / "config_d.yaml", "r") as f:
        config = yaml.safe_load(f)
    if (root_dir / "config.yaml").is_file():
        with open(root_dir / "config.yaml", "r") as f:
            config |= yaml.safe_load(f)  # Overwrite default config with keys from user config
    loglvl = config["loglevel"].lower()
    if loglvl == "debug":
        config["loglevel"] = logging.DEBUG
    elif loglvl == "info":
        config["loglevel"] = logging.INFO
    elif loglvl == "warning":
        config["loglevel"] = logging.WARNING
    elif loglvl == "error":
        config["loglevel"] = logging.ERROR
    else:
        raise RuntimeError(f"Loglevel {config['loglevel']} in config not supported!")
    return SimpleNamespace(**config)


if __name__ == "__main__":
    config = load_config()
    logging.basicConfig(level=logging.INFO)
    logging.getLogger("soulsai").setLevel(config.loglevel)

    # Enable training interrupt with 'Enter' key
    stop_flag = [False]

    def exit_callback():
        stop_flag[0] = True

    keyboard.add_hotkey("enter", exit_callback)

    # Connector enables non-blocking server interaction
    con = Connector()
    env = gym.make("SoulsGymIudex-v0")
    # Training setup
    n_actions = env.action_space.n
    gamestate = GameState()
    gamestate.player_max_hp = 1
    gamestate.player_max_sp = 1
    gamestate.boss_max_hp = 1
    n_states = len(gamestate2np(gamestate))

    logger.info("Press 'Enter' to end training")
    try:
        while not stop_flag[0]:
            state = env.reset()
            done = False
            total_reward = 0.
            steps = 1
            while not done and not stop_flag[0]:
                state_A = gamestate2np(state)
                with con:
                    eps = con.eps
                    model_id = con.model_id
                    if np.random.rand() < eps:
                        action = env.action_space.sample()
                    else:
                        action = con.agent(state_A)
                next_state, reward, done, _ = env.step(action)
                con.push_sample(model_id, [state, action, reward, next_state, done])
                state = next_state
                total_reward += reward
                steps += 1
            boss_hp = state.boss_hp
            win = boss_hp == 0
            con.push_telemetry(total_reward, steps, boss_hp, win, eps)
        logger.info("Exiting training")
    finally:
        env.close()
        con.close()
