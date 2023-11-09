"""This script is the main function for starting the sampling client.

Clients read the local configuration, connect to the training server, and download the current
training configuration. Clients then enter the main training loop and start to generate samples from
the environment.

Note:
    If you train on SoulsGym, the player has to be loaded into the game in order to start training.
    Also remember to not interact with your keyboard in any way during training with SoulsGym.

Example:
    To start training on a client, make sure you have configured the correct Redis address in the
    config files, and the server is running. You can then start the training by running the script
    from the package root folder:

        $ python soulsai/distributed/client/main.py
"""
from pathlib import Path
import logging

import soulsgym  # noqa: F401, needs to register SoulsGym envs with gym module

from soulsai.distributed.client.dqn_client import dqn_client
from soulsai.distributed.client.ppo_client import ppo_client
from soulsai.utils import load_config
from soulsai.exception import InvalidConfigError
from soulsai.utils import load_remote_config, load_redis_secret

if __name__ == "__main__":
    logging.basicConfig()
    node_dir = Path(__file__).parents[3] / "config"
    local_config = load_config(node_dir / "config_d.yaml", node_dir / "config.yaml")
    secret = load_redis_secret(Path(__file__).parents[3] / "config/secrets/redis.secret")
    config = load_remote_config(local_config.redis_address, secret, local_config=local_config.local)
    if config.algorithm.lower() == "dqn":
        dqn_client(config)
    elif config.algorithm.lower() == "ppo":
        ppo_client(config)
    else:
        raise InvalidConfigError(f"Algorithm type {config.algorithm} is not supported")
