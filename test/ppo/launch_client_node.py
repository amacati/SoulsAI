from pathlib import Path

import numpy as np

from soulsai.distributed.client.ppo_client import ppo_client
from soulsai.utils import load_config


def tel_callback(total_reward, steps, state):
    return total_reward, steps, bool(total_reward > 200)


def encode_tel(msg):
    return {"reward": msg[1], "steps": msg[2], "boss_hp": 0, "win": msg[3], "eps": 0}


def encode_sample(msg):
    msg = msg[4]
    msg[0] = np.float64(msg[0])
    return [list(msg[0]), msg[1], msg[2], msg[3], msg[4]]


if __name__ == "__main__":
    root_dir = Path(__file__).parents[1]
    config = load_config(root_dir / "common" / "config_d.yaml", root_dir / "ppo" / "config.yaml")
    ppo_client(config, tf_state_callback=lambda x: x, tel_callback=tel_callback,
               encode_sample=encode_sample, encode_tel=encode_tel)
