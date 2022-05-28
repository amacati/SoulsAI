import logging

import gym
import soulsgym  # noqa: F401

from dummy_api import connect_to_server


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    env = gym.make("SoulsGymIudex-v0")

    server = connect_to_server()
    model_id = server.sub_msgs[1]
    server.sub_msgs[0] = False
    try:
        for _ in range(3):
            state = env.reset()
            done = False
            while not done:
                next_state, _, done, _ = env.step(19)
                server.push_sample({"model_id": model_id, "sample": (state, next_state, done)})
                state = next_state
                if server.sub_msgs[0]:
                    model_id = server.sub_msgs[1]
                    server.sub_msgs[0] = False
    finally:
        env.close()
