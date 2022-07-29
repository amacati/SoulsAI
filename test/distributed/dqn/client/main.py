import logging
from pathlib import Path
from types import SimpleNamespace

import yaml
import numpy as np
import gym

from test.distributed.dqn.client.connector import Connector

logger = logging.getLogger(__name__)


def load_config():
    root_dir = Path(__file__).parent
    with open(root_dir / "config_d.yaml", "r") as f:
        config = yaml.safe_load(f)
    if (root_dir / "config.yaml").is_file():
        with open(root_dir / "config.yaml", "r") as f:
            _config = yaml.safe_load(f)  # Overwrite default config with keys from user config
        if _config is not None:
            config |= _config
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

    con = Connector()
    env = gym.make("LunarLander-v2")

    try:
        for i in range(config.nepisodes):
            state = env.reset()
            done = False
            total_reward = 0.
            steps = 1
            while not done:
                with con:
                    eps = con.eps
                    model_id = con.model_id
                    if np.random.rand() < eps:
                        action = env.action_space.sample()
                    else:
                        action = con.agent(state)
                next_state, reward, done, _ = env.step(action)
                con.push_sample(model_id, [state, action, reward, next_state, done])
                state = next_state
                total_reward += reward
                steps += 1
            win = total_reward > 200
            con.push_telemetry(total_reward, steps, win, eps)
        logger.info("Exiting training")
    finally:
        env.close()
        con.close()
