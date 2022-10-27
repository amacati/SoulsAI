from pathlib import Path
from soulsai.utils import load_remote_config, load_redis_secret

import soulsgym  # noqa: F401, needs to register SoulsGym envs with gym module

from soulsai.distributed.client.dqn_client import dqn_client
from soulsai.distributed.client.ppo_client import ppo_client
from soulsai.utils import load_config
from soulsai.data.transformation import GameStateTransformer
from soulsai.exception import InvalidConfigError


def tel_callback(total_reward, steps, state, eps):
    return total_reward, steps, state.boss_hp, state.boss_hp == 0, eps


def dqn_encode_sample(msg):
    return [msg[2][0].tolist(), msg[2][1], msg[2][2], msg[2][3].tolist(), msg[2][4]]


def ppo_encode_sample(msg):
    return [msg[4][0].tolist(), msg[4][1], msg[4][2], msg[4][3], msg[4][4]]


def encode_tel(msg):
    return {"reward": msg[1], "steps": msg[2], "boss_hp": msg[3], "win": msg[4], "eps": msg[5]}


if __name__ == "__main__":
    node_dir = Path(__file__).parents[3] / "config"
    config = load_config(node_dir / "config_d.yaml", node_dir / "config.yaml")
    secret = load_redis_secret(Path(__file__).parents[3] / "config" / "redis.secret")
    config = load_remote_config(config.redis_address, secret)
    tf_transformer = GameStateTransformer()
    if config.algorithm.lower() == "dqn":
        dqn_client(config, tf_state_callback=tf_transformer.transform, tel_callback=tel_callback,
                   encode_sample=dqn_encode_sample, encode_tel=encode_tel,
                   episode_end_callback=tf_transformer.reset)
    elif config.algorithm.lower() == "ppo":
        ppo_client(config, tf_state_callback=tf_transformer.transform, tel_callback=tel_callback,
                   encode_sample=ppo_encode_sample, encode_tel=encode_tel,
                   episode_end_callback=tf_transformer.reset)
    else:
        raise InvalidConfigError(f"Algorithm type {config.algorithm} is not supported")
