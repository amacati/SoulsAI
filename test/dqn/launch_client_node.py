"""Launch script for the test version of the DQN training client compatible with LunarLander-v2."""
from pathlib import Path

from soulsai.distributed.client.dqn_client import dqn_client
from soulsai.utils import load_config

if __name__ == "__main__":
    root_dir = Path(__file__).parents[1]
    config = load_config(root_dir / "common" / "config_d.yaml", root_dir / "dqn" / "config.yaml")
    dqn_client(config)
