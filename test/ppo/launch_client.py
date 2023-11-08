"""Launch script for the test version of the PPO training client compatible with LunarLander-v2."""
from pathlib import Path

from soulsai.distributed.client.ppo_client import ppo_client
from soulsai.utils import load_config, load_remote_config, load_redis_secret

if __name__ == "__main__":
    root_dir = Path(__file__).parents[2] / "config"
    local_config = load_config(root_dir / "config_d.yaml", root_dir / "config.yaml")
    secret = load_redis_secret(Path("/run/secrets/redis_secret"))
    config = load_remote_config(local_config.redis_address, secret, local_config=local_config.local)
    ppo_client(config)
