"""The DQN client module contains the sampling loop for DQN on worker nodes.

If specified, the training function is executed by a :class:`.ClientWatchdog` to restart the
sampling whenever the sampling rate drops below the expected value.
"""
import logging
from threading import Event
import time
from collections import deque
from types import SimpleNamespace
from typing import Callable
from multiprocessing.sharedctypes import Synchronized

import numpy as np
import gym
from soulsai.core.noise import get_noise_class
from soulsai.utils import namespace2dict
from soulsai.distributed.client.connector import DQNConnector
from soulsai.distributed.client.watchdog import ClientWatchdog

logger = logging.getLogger(__name__)


def dqn_client(config: SimpleNamespace,
               tf_state_callback: Callable,
               encode_sample: Callable,
               encode_tel: Callable,
               episode_end_callback: Callable | None = None):
    """Wrap the the DQN client main function and automatically parameterize it.

    If the training is configured to use a watchdog, the watchdog starts the training and adds
    additional arguments to check the health of the training process by measuring the sampling rate.

    Args:
        config: The training configuration.
        tf_state_callback: Callback to transform environment observations into agent inputs.
        encode_sample: Function to encode sample messages for redis.
        encode_tel: Function to encode the telemetry information for redis.
        episode_end_callback: Callback for functions that should be called at the end of an episode.
    """
    logging.basicConfig(level=config.loglevel)
    logging.getLogger("soulsai").setLevel(config.loglevel)
    logger.info("Launching DQN client")

    if config.watchdog.enable:
        minimum_samples_per_minute = config.watchdog.minimum_samples
        external_args = (config, tf_state_callback, encode_sample, encode_tel, episode_end_callback)
        watchdog = ClientWatchdog(_dqn_client, minimum_samples_per_minute, external_args)
        watchdog.start()
    else:
        _dqn_client(config, tf_state_callback, encode_sample, encode_tel, episode_end_callback)


def _dqn_client(config: SimpleNamespace,
                tf_state_callback: Callable,
                encode_sample: Callable,
                encode_tel: Callable,
                episode_end_callback: Callable | None = None,
                stop_flag: Event = Event(),
                sample_gauge: Synchronized | None = None):
    """DQN client main function.

    Data processing, sample encodings etc. are configurable to customize the client for different
    environments. Messages to the redis client have to be encoded since redis messages can't use
    arbitrary data types.

    Args:
        config: The training configuration.
        tf_state_callback: Callback to transform environment observations into agent inputs.
        encode_sample: Function to encode sample messages for redis.
        encode_tel: Function to encode the telemetry information for redis.
        episode_end_callback: Callback for functions that should be called at the end of an episode.
        stop_flag: Event flag to stop the training.
        sample_gauge: Optional parameter that allows external processes to measure the current
            sample rate.
    """
    if config.enable_interrupt:
        import keyboard  # Keyboard should not be imported in Docker during testing

        def exit_callback():
            stop_flag.set()

        keyboard.add_hotkey("enter", exit_callback)
        logger.info("Press 'Enter' to end training")

    # DQNConnector enables non-blocking server interaction
    con = DQNConnector(config)
    env_kwargs = namespace2dict(config.env_kwargs) if config.use_env_kwargs else {}
    env = gym.make(config.env, **env_kwargs)
    noise = get_noise_class(config.dqn.noise)(**namespace2dict(config.dqn.noise_kwargs))
    logger.info("Client node running")
    try:
        episode_id = 0
        states = deque(maxlen=config.dqn.multistep + 1)
        actions = deque(maxlen=config.dqn.multistep)
        rewards = deque(maxlen=config.dqn.multistep)
        infos = deque(maxlen=config.dqn.multistep)
        while (not stop_flag.is_set() and episode_id != config.max_episodes and  # noqa: W504
               not con.shutdown.is_set()):
            episode_id += 1
            state = tf_state_callback(env.reset())
            if config.dqn.action_masking:
                valid_actions = env.current_valid_actions()
                action_mask = np.zeros(config.n_actions)
                action_mask[valid_actions] = 1
            if sample_gauge and episode_id == 1:
                current_gauge_start_time = time.time()
                current_gauge_cnt = 0
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
                    else:
                        state_n = con.normalizer.normalize(state) if config.dqn.normalize else state
                        if config.dqn.action_masking:
                            action = con.agent(state_n, action_mask)
                        else:
                            action = con.agent(state_n)
                next_state, reward, done, info = env.step(action)
                next_state = tf_state_callback(next_state)
                states.append(next_state)
                actions.append(action)
                rewards.append(reward)
                infos.append(info)
                total_reward += reward
                if len(rewards) == config.dqn.multistep:
                    sum_r = sum([rewards[i] * config.gamma**i for i in range(config.dqn.multistep)])
                    con.push_sample(
                        model_id,
                        encode_sample(states[0], actions[0], sum_r, states[-1], done, infos[-1]))
                    if sample_gauge:
                        current_gauge_cnt += 1
                        tnow = time.time()
                        if tnow - current_gauge_start_time > 10:
                            td = tnow - current_gauge_start_time
                            sample_gauge.value = int(current_gauge_cnt * 60 / td)
                            current_gauge_cnt = 0
                            current_gauge_start_time = tnow
                state = next_state
                if config.dqn.action_masking:
                    action_mask[:] = 0
                    action_mask[info["allowed_actions"]] = 1
                steps += 1
                if config.step_delay:  # Enable Dockerfiles to simulate slow clients
                    time.sleep(config.step_delay)
            if not stop_flag.is_set():
                for i in range(1, len(rewards)):
                    sum_r = sum(
                        [rewards[i + j] * config.gamma**j for j in range(config.dqn.multistep - i)])
                    con.push_sample(
                        model_id,
                        encode_sample(states[i], actions[i], sum_r, states[-1], done, infos[-1]))
                    if sample_gauge:
                        current_gauge_cnt += 1
                        tnow = time.time()
                        if tnow - current_gauge_start_time > 10:
                            td = tnow - current_gauge_start_time
                            sample_gauge.value = int(current_gauge_cnt * 60 / td)
                            current_gauge_cnt = 0
                            current_gauge_start_time = tnow
                noise.reset()
                con.push_telemetry(encode_tel(total_reward, steps, state, eps))
            if episode_end_callback is not None:
                episode_end_callback()
        logger.info("Exiting training")
    finally:
        env.close()
        con.close()
