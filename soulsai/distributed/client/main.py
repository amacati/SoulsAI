from pathlib import Path

import soulsgym  # noqa: F401, needs to register SoulsGym envs with gym module

from soulsai.distributed.client.client_node import client_node
from soulsai.utils import load_config
from soulsai.core.utils import gamestate2np


def tel_callback(total_reward, steps, state, eps):
    return total_reward, steps, state.boss_hp, state.boss_hp == 0, eps


if __name__ == "__main__":
    node_dir = Path(__file__).parents[3] / "config"
    config = load_config(node_dir / "config_d.yaml", node_dir / "config.yaml")
    client_node(config, tf_state_callback=gamestate2np, tel_callback=tel_callback)
