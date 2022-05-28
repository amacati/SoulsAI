import logging
import random
import time

import gym
import json
import soulsgym  # noqa: F401
import redis


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    env = gym.make("SoulsGymIudex-v0")
    update_flag = [False]

    def model_update_callback(_):
        update_flag[0] = True

    red = redis.Redis(host='localhost', port=6379, db=0, charset="utf-8", decode_responses=True)

    model_id = red.get("model_id")
    pubsub = red.pubsub()
    pubsub.psubscribe(**{"model_update": model_update_callback})
    pubsub.run_in_thread(sleep_time=.01, daemon=True)

    try:
        for _ in range(3):
            state = env.reset()
            state = random.random()
            done = False
            while not done:
                next_state, _, done, _ = env.step(19)
                sample = json.dumps({"model_id": model_id, "sample": [state, next_state, done]})
                red.publish("samples", sample)
                state = next_state
                if update_flag[0]:
                    update_flag[0] = False
                    model_id = red.get("model_id")
                    model = red.get(model_id)
    finally:
        env.close()
        ...
