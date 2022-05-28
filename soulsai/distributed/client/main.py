import logging
import json

import numpy as np
import gym
import soulsgym  # noqa: F401
from soulsgym.core.game_state import GameState
import redis

from soulsai.core.agent import ClientAgent
from soulsai.core.utils import gamestate2np


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

    # Training setup
    n_actions = env.action_space.n
    gamestate = GameState()
    gamestate.player_max_hp = 1
    gamestate.player_max_sp = 1
    gamestate.boss_max_hp = 1
    n_states = len(gamestate2np(gamestate))
    agent = ClientAgent(n_states, n_actions)
    model_params = red.get(model_id)
    agent.from_json(model_params["model"])
    eps = model_params["eps"]

    try:
        for _ in range(3):
            state = env.reset()
            done = False
            while not done:
                state_A = gamestate2np(state)
                action = env.action_space.sample() if np.random.rand() < eps else agent(state_A)
                next_state, reward, done, _ = env.step(action)
                sample = [state.as_json(), action, reward, next_state.as_json(), done]
                red.publish("samples", json.dumps({"model_id": model_id, "sample": sample}))
                state = next_state
                if update_flag[0]:
                    update_flag[0] = False
                    model_id = red.get("model_id")
                    model_params = red.get(model_id)
                    agent.from_json(model_params["model"])
                    eps = model_params["eps"]
    finally:
        env.close()
        ...
