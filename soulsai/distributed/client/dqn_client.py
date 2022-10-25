import logging
from threading import Event
import time
from collections import deque

import numpy as np
import gym
from soulsai.core.noise import UniformDiscreteNoise
from soulsai.exception import InvalidConfigError

from soulsai.distributed.client.connector import DQNConnector

logger = logging.getLogger(__name__)


def dqn_client(config, tf_state_callback, tel_callback, encode_sample, encode_tel):
    logging.basicConfig(level=config.loglevel)
    logging.getLogger("soulsai").setLevel(config.loglevel)
    logger.info("Launching DQN client")

    stop_flag = Event()
    if config.enable_interrupt:
        import keyboard  # Keyboard should not be imported in Docker during testing

        def exit_callback():
            stop_flag.set()

        keyboard.add_hotkey("enter", exit_callback)
        logger.info("Press 'Enter' to end training")

    # DQNConnector enables non-blocking server interaction
    con = DQNConnector(config, encode_sample, encode_tel)
    env = gym.make(config.env)
    noise = _get_noise(config)

    logger.info("Client node running")
    try:
        episode_id = 0
        states = deque(maxlen=config.dqn_multistep + 1)
        actions = deque(maxlen=config.dqn_multistep)
        rewards = deque(maxlen=config.dqn_multistep)
        while not stop_flag.is_set() and episode_id != config.max_episodes:
            episode_id += 1
            state = env.reset()
            done = False
            total_reward = 0.
            steps = 1
            states.clear()
            actions.clear()
            rewards.clear()
            states.append(state)
            while not done and not stop_flag.is_set():
                tfstate = tf_state_callback(state)
                with con:
                    eps = con.eps
                    model_id = con.model_id
                    if np.random.rand() < eps:
                        action = noise.sample()
                    else:
                        action = con.agent(tfstate)
                next_state, reward, done, _ = env.step(action)
                states.append(next_state)
                actions.append(action)
                rewards.append(reward)
                total_reward += reward
                if len(rewards) == config.dqn_multistep:
                    sum_r = sum([rewards[i] * config.gamma**i for i in range(config.dqn_multistep)])
                    con.push_sample(model_id, [states[0], actions[0], sum_r, states[-1], done])
                state = next_state
                steps += 1
                if config.step_delay:  # Enable Dockerfiles to simulate slow clients
                    time.sleep(config.step_delay)
            if not stop_flag.is_set():
                for i in range(1, len(rewards)):
                    sum_r = sum([rewards[i + j] * config.gamma**j for j in range(config.dqn_multistep - i)])
                    con.push_sample(model_id, [states[i], actions[i], sum_r, states[-1], done])
                noise.reset()
                con.push_telemetry(*tel_callback(total_reward, steps, state, eps))
        logger.info("Exiting training")
    finally:
        env.close()
        con.close()


def _get_noise(config):
    if config.noise == "UniformDiscreteNoise":
        noise_cls = UniformDiscreteNoise
    else:
        raise InvalidConfigError(f"Noise type {config.noise} not supported.")
    return noise_cls(**config.noise_kwargs)
