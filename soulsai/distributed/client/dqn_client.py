import logging
from threading import Event
import time
from collections import deque

import numpy as np
import gym
from soulsai.core.noise import UniformDiscreteNoise
from soulsai.utils import namespace2dict
from soulsai.distributed.client.connector import DQNConnector
from soulsai.exception import InvalidConfigError

logger = logging.getLogger(__name__)


def dqn_client(config, tf_state_callback, tel_callback, encode_sample, encode_tel,
               episode_end_callback=None):
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
        states = deque(maxlen=config.dqn.multistep + 1)
        actions = deque(maxlen=config.dqn.multistep)
        rewards = deque(maxlen=config.dqn.multistep)
        while not stop_flag.is_set() and episode_id != config.max_episodes:
            episode_id += 1
            state = tf_state_callback(env.reset())
            done = False
            total_reward = 0.
            steps = 1
            states.clear()
            actions.clear()
            rewards.clear()
            states.append(state)
            while not done and not stop_flag.is_set():
                with con:
                    eps = con.eps
                    model_id = con.model_id
                    if np.random.rand() < eps:
                        action = noise.sample()
                    else:
                        action = con.agent(state)
                next_state, reward, done, _ = env.step(action)
                next_state = tf_state_callback(next_state)
                states.append(next_state)
                actions.append(action)
                rewards.append(reward)
                total_reward += reward
                if len(rewards) == config.dqn.multistep:
                    sum_r = sum([rewards[i] * config.gamma**i for i in range(config.dqn.multistep)])
                    con.push_sample(model_id, [states[0], actions[0], sum_r, states[-1], done])
                state = next_state
                steps += 1
                if config.step_delay:  # Enable Dockerfiles to simulate slow clients
                    time.sleep(config.step_delay)
            if not stop_flag.is_set():
                for i in range(1, len(rewards)):
                    sum_r = sum([rewards[i + j] * config.gamma**j for j in range(config.dqn.multistep - i)])  # noqa: E501
                    con.push_sample(model_id, [states[i], actions[i], sum_r, states[-1], done])
                noise.reset()
                con.push_telemetry(*tel_callback(total_reward, steps, state, eps))
            if episode_end_callback is not None:
                episode_end_callback()
        logger.info("Exiting training")
    finally:
        env.close()
        con.close()


def _get_noise(config):
    if config.dqn.noise == "UniformDiscreteNoise":
        noise_cls = UniformDiscreteNoise
    else:
        raise InvalidConfigError(f"Noise type {config.dqn.noise} not supported.")
    return noise_cls(**namespace2dict(config.dqn.noise_kwargs))
