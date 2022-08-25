from pathlib import Path
from soulsai.utils import load_remote_config, load_redis_secret

import soulsgym  # noqa: F401, needs to register SoulsGym envs with gym module

from soulsai.distributed.client.client_node import client_node
from soulsai.utils import load_config
from soulsai.core.utils import gamestate2np


def tel_callback(total_reward, steps, state, eps):
    return total_reward, steps, state.boss_hp, state.boss_hp == 0, eps


def encode_sample(msg):
    return [msg[2][0].as_json(), msg[2][1], msg[2][2], msg[2][3].as_json(), msg[2][4]]


def encode_tel(msg):
    return {"reward": msg[1], "steps": msg[2], "boss_hp": msg[3], "win": msg[4], "eps": msg[5]}


if __name__ == "__main__":
    node_dir = Path(__file__).parents[3] / "config"
    config = load_config(node_dir / "config_d.yaml", node_dir / "config.yaml")
    secret = load_redis_secret(Path(__file__).parents[3] / "config" / "redis.secret")
    config = load_remote_config(config.redis_address, secret)
    client_node(config, tf_state_callback=gamestate2np, tel_callback=tel_callback,
                encode_sample=encode_sample, encode_tel=encode_tel)
