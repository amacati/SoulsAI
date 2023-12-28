"""Script to issue a quicksave command to the training node.

The script has to be run on the server that is running the Redis node. The checkpoint will be put
into a separate folder.
"""
from pathlib import Path

from redis import Redis

from soulsai.utils import load_redis_secret


if __name__ == "__main__":
    config_dir = Path(__file__).parents[1] / "config"
    secret = load_redis_secret(config_dir / "secrets/redis.secret")
    red = Redis(host="localhost", port=6379, password=secret, db=0, decode_responses=True)
    red.publish("manual_save", 1)
