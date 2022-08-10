import logging
from threading import Event
import time

import numpy as np
import gym

from soulsai.distributed.client.connector import Connector

logger = logging.getLogger(__name__)


def client_node(config, tf_state_callback, tel_callback, encode_sample, encode_tel):
    logging.basicConfig(level=config.loglevel)
    logging.getLogger("soulsai").setLevel(config.loglevel)

    stop_flag = Event()
    if config.enable_interrupt:
        import keyboard  # Keyboard should not be imported in Docker during testing

        def exit_callback():
            stop_flag.set()

        keyboard.add_hotkey("enter", exit_callback)
        logger.info("Press 'Enter' to end training")

    # Connector enables non-blocking server interaction
    con = Connector(config, encode_sample, encode_tel)
    env = gym.make(config.env)

    logger.info("Client node running")
    try:
        episode_id = 0
        while not stop_flag.is_set() and episode_id != config.max_episodes:
            episode_id += 1
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
                if config.step_delay:  # Enable Dockerfiles to simulate slow clients
                    time.sleep(config.step_delay)
            con.push_telemetry(*tel_callback(total_reward, steps, state, eps))
        logger.info("Exiting training")
    finally:
        env.close()
        con.close()
