import logging
from threading import Event
import time
from collections import deque

import numpy as np
import gym
from soulsai.core.noise import UniformDiscreteNoise, MaskedDiscreteNoise
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
    env_kwargs = namespace2dict(config.env_kwargs) if config.use_env_kwargs else {}
    env = gym.make(config.env, **env_kwargs)
    noise = _get_noise(config)

    logger.info("Client node running")
    try:
        episode_id = 0
        states = deque(maxlen=config.dqn.multistep + 1)
        actions = deque(maxlen=config.dqn.multistep)
        rewards = deque(maxlen=config.dqn.multistep)
        infos = deque(maxlen=config.dqn.multistep)
        while not stop_flag.is_set() and episode_id != config.max_episodes:
            episode_id += 1
            state = tf_state_callback(env.reset())
            if config.dqn.action_masking:
                valid_actions = env.current_valid_actions()
                action_mask = np.zeros(config.n_actions)
                action_mask[valid_actions] = 1
            done = False
            total_reward = 0.
            steps = 1
            states.clear()
            actions.clear()
            rewards.clear()
            infos.clear()
            states.append(state)
            while not done and not stop_flag.is_set():
                with con:
                    eps = con.eps
                    model_id = con.model_id
                    if np.random.rand() < eps:
                        if config.dqn.noise == "MaskedDiscreteNoise":
                            action = noise.sample(action_mask)
                        else:
                            action = noise.sample()
                    elif config.dqn.action_masking:
                        action = con.agent(state, action_mask)
                    else:
                        action = con.agent(state)
                next_state, reward, done, info = env.step(action)
                next_state = tf_state_callback(next_state)
                states.append(next_state)
                actions.append(action)
                rewards.append(reward)
                infos.append(info)
                total_reward += reward
                if len(rewards) == config.dqn.multistep:
                    sum_r = sum([rewards[i] * config.gamma**i for i in range(config.dqn.multistep)])
                    sample = [states[0], actions[0], sum_r, states[-1], done, infos[-1]]
                    con.push_msg("sample", model_id, sample)
                state = next_state
                if config.dqn.action_masking:
                    action_mask[:] = 0
                    action_mask[info["allowed_actions"]] = 1
                steps += 1
                if config.step_delay:  # Enable Dockerfiles to simulate slow clients
                    time.sleep(config.step_delay)
            if not stop_flag.is_set():
                for i in range(1, len(rewards)):
                    sum_r = sum([rewards[i + j] * config.gamma**j for j in range(config.dqn.multistep - i)])  # noqa: E501
                    sample = [states[i], actions[i], sum_r, states[-1], done, infos[-1]]
                    con.push_msg("sample", model_id, sample)
                noise.reset()
                con.push_msg("telemetry", *tel_callback(total_reward, steps, state, eps))
            if episode_end_callback is not None:
                episode_end_callback()
        logger.info("Exiting training")
    finally:
        env.close()
        con.close()


def _get_noise(config):
    if config.dqn.noise == "UniformDiscreteNoise":
        noise_cls = UniformDiscreteNoise
    elif config.dqn.noise == "MaskedDiscreteNoise":
        noise_cls = MaskedDiscreteNoise
    else:
        raise InvalidConfigError(f"Noise type {config.dqn.noise} not supported.")
    return noise_cls(**namespace2dict(config.dqn.noise_kwargs))
