"""Launch script for the test version of the Atari Pong DQN training client."""
from pathlib import Path

from soulsai.distributed.client.dqn_client import dqn_client
from soulsai.utils import load_config

if __name__ == "__main__":
    root_dir = Path(__file__).parents[2] / "config"
    config = load_config(root_dir / "config_d.yaml", root_dir / "config.yaml")
    dqn_client(config)
