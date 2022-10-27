from pathlib import Path

import numpy as np

from soulsai.distributed.client.dqn_client import dqn_client
from soulsai.utils import load_config


def tel_callback(total_reward, steps, state, eps):
    return total_reward, steps, bool(total_reward > 200), eps


def encode_tel(msg):
    return {"reward": msg[1], "steps": msg[2], "boss_hp": 0, "win": msg[3], "eps": msg[4]}


def encode_sample(msg):
    msg[2][0], msg[2][3] = np.float64(msg[2][0]), np.float64(msg[2][3])
    return [list(msg[2][0]), msg[2][1], msg[2][2], list(msg[2][3]), msg[2][4]]


if __name__ == "__main__":
    root_dir = Path(__file__).parents[1]
    config = load_config(root_dir / "common" / "config_d.yaml", root_dir / "dqn" / "config.yaml")
    dqn_client(config, tf_state_callback=lambda x: x, tel_callback=tel_callback,
               encode_sample=encode_sample, encode_tel=encode_tel)
