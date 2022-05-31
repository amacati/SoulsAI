import logging
import json
from pathlib import Path

import numpy as np
import gym
import soulsgym  # noqa: F401
from soulsgym.core.game_state import GameState
import redis

from soulsai.core.agent import ClientAgent
from soulsai.core.utils import gamestate2np


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    update_flag = [False]

    def model_update_callback(_):
        update_flag[0] = True

    with open(Path(__file__).parent / "redis.secret") as f:
        conf = f.readlines()
    secret = None
    for line in conf:
        if len(line) > 12 and line[0:12] == "requirepass ":
            secret = line[12:]
            break
    if secret is None:
        raise RuntimeError("Missing password configuration for redis in redis.secret")

    red = redis.Redis(host='localhost', password=secret, port=6379, db=0)
    model_id = red.get("model_id").decode("utf-8")
    pubsub = red.pubsub()
    pubsub.psubscribe(model_update=model_update_callback)
    pubsub.run_in_thread(sleep_time=.01, daemon=True)

    env = gym.make("SoulsGymIudex-v0")

    # Training setup
    n_actions = env.action_space.n
    gamestate = GameState()
    gamestate.player_max_hp = 1
    gamestate.player_max_sp = 1
    gamestate.boss_max_hp = 1
    n_states = len(gamestate2np(gamestate))
    agent = ClientAgent(n_states, n_actions)
    model_params = red.hgetall(model_id)
    model_params = {key.decode("utf-8"): value for key, value in model_params.items()}
    agent.deserialize(model_params)
    eps = float(model_params["eps"].decode("utf-8"))

    try:
        while True:
            state = env.reset()
            done = False
            reward = 0.
            steps = 1
            while not done:
                state_A = gamestate2np(state)
                action = env.action_space.sample() if np.random.rand() < eps else agent(state_A)
                next_state, reward, done, _ = env.step(action)
                sample = [state.as_json(), action, reward, next_state.as_json(), done]
                red.publish("samples", json.dumps({"model_id": model_id, "sample": sample}))
                state = next_state
                if update_flag[0]:
                    update_flag[0] = False
                    model_id = red.get("model_id").decode("utf-8")
                    model_params = red.hgetall(model_id)
                    model_params = {key.decode("utf-8"): value for key, value in model_params.items()}
                    agent.deserialize(model_params)
                    eps = float(model_params["eps"].decode("utf-8"))
            boss_hp = state.boss_hp
            win = boss_hp == 0
            red.publish("telemetry", json.dumps({"reward": reward, "steps": steps,
                                                 "boss_hp": boss_hp, "win": win}))
    finally:
        env.close()
        ...
