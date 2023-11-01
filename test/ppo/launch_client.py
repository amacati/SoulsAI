"""Launch script for the test version of the PPO training client compatible with LunarLander-v2."""
from pathlib import Path

from soulsai.distributed.client.ppo_client import ppo_client
from soulsai.utils import load_config

if __name__ == "__main__":
    root_dir = Path(__file__).parents[2] / "config"
    config = load_config(root_dir / "config_d.yaml", root_dir / "config.yaml")
    ppo_client(config)
