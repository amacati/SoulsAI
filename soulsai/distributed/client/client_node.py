import logging
from threading import Event

import numpy as np
import gym
import keyboard

from soulsai.distributed.client.connector import Connector

logger = logging.getLogger(__name__)


def client_node(config, tf_state_callback, tel_callback):
    logging.basicConfig(level=config.loglevel)
    logging.getLogger("soulsai").setLevel(config.loglevel)

    # Enable training interrupt with 'Enter' key
    stop_flag = Event()

    def exit_callback():
        stop_flag.set()

    keyboard.add_hotkey("enter", exit_callback)

    # Connector enables non-blocking server interaction
    con = Connector(config)
    env = gym.make(config.env)

    logger.info("Press 'Enter' to end training")
    try:
        while not stop_flag.is_set():
            state = env.reset()
            done = False
            total_reward = 0.
            steps = 1
            while not done and not stop_flag.is_set():
                tfstate = tf_state_callback(state)
                with con:
                    eps = con.eps
                    model_id = con.model_id
                    if np.random.rand() < eps:
                        action = env.action_space.sample()
                    else:
                        action = con.agent(tfstate)
                next_state, reward, done, _ = env.step(action)
                con.push_sample(model_id, [state, action, reward, next_state, done])
                state = next_state
                total_reward += reward
                steps += 1
            con.push_telemetry(*tel_callback(total_reward, steps, state, eps))
        logger.info("Exiting training")
    finally:
        env.close()
        con.close()
