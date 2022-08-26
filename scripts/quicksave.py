from pathlib import Path

from redis import Redis

from soulsai.utils import load_config, load_redis_secret


def main():
    config_dir = Path(__file__).parents[1] / "config"
    config = load_config(config_dir / "config_d.yaml", config_dir / "config.yaml")
    secret = load_redis_secret(config_dir / "redis.secret")
    red = Redis(host=config.redis_address, port=6379, password=secret, db=0, decode_responses=True)
    red.publish("manual_save", 1)


if __name__ == "__main__":
    main()
