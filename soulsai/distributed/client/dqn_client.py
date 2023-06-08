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
import gymnasium as gym
from soulsai.core.noise import get_noise_class
from soulsai.utils import namespace2dict
from soulsai.distributed.common.serialization import DQNSerializer
from soulsai.distributed.client.connector import DQNConnector
from soulsai.distributed.client.watchdog import ClientWatchdog

logger = logging.getLogger(__name__)


def dqn_client(config: SimpleNamespace,
               tf_obs_callback: Callable,
               episode_end_callback: Callable | None = None):
    """Wrap the the DQN client main function and automatically parameterize it.

    If the training is configured to use a watchdog, the watchdog starts the training and adds
    additional arguments to check the health of the training process by measuring the sampling rate.

    Args:
        config: The training configuration.
        tf_obs_callback: Callback to transform environment observations into agent inputs.
        episode_end_callback: Callback for functions that should be called at the end of an episode.
    """
    logging.basicConfig(level=config.loglevel)
    logging.getLogger("soulsai").setLevel(config.loglevel)
    logger.info("Launching DQN client")

    if config.watchdog.enable:
        minimum_samples_per_minute = config.watchdog.minimum_samples
        external_args = (config, tf_obs_callback, episode_end_callback)
        watchdog = ClientWatchdog(_dqn_client, minimum_samples_per_minute, external_args)
        watchdog.start()
    else:
        _dqn_client(config, tf_obs_callback, episode_end_callback)


def _dqn_client(config: SimpleNamespace,
                tf_obs_callback: Callable,
                episode_end_callback: Callable | None = None,
                stop_flag: Event = Event(),
                sample_gauge: Synchronized | None = None):
    """DQN client main function.

    Data processing, sample encodings etc. are configurable to customize the client for different
    environments. Messages to the redis client have to be encoded since redis messages can't use
    arbitrary data types.

    Args:
        config: The training configuration.
        tf_obs_callback: Callback to transform environment observations into agent inputs.
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
    serializer = DQNSerializer(env_id=config.env)
    noise = get_noise_class(config.dqn.noise)(**namespace2dict(config.dqn.noise_kwargs))
    logger.info("Client node running")
    try:
        episode_id = 0
        observations = deque(maxlen=config.dqn.multistep + 1)
        actions = deque(maxlen=config.dqn.multistep)
        rewards = deque(maxlen=config.dqn.multistep)
        infos = deque(maxlen=config.dqn.multistep)
        while (not stop_flag.is_set() and episode_id != config.max_episodes and  # noqa: W504
               not con.shutdown.is_set()):
            episode_id += 1
            obs, info = env.reset()
            obs = tf_obs_callback(obs)
            if config.dqn.action_masking:
                action_mask = np.zeros(config.n_actions)
                action_mask[info["allowed_actions"]] = 1
            if sample_gauge and episode_id == 1:
                current_gauge_start_time = time.time()
                current_gauge_cnt = 0
            terminated = False
            total_reward = 0.
            steps = 1
            observations.clear()
            actions.clear()
            rewards.clear()
            infos.clear()
            observations.append(obs)
            while not terminated and not stop_flag.is_set():
                with con:
                    eps = con.eps
                    model_id = con.model_id
                    if np.random.rand() < eps:
                        if config.dqn.noise == "MaskedDiscreteNoise":
                            action = noise.sample(action_mask)
                        else:
                            action = noise.sample()
                    else:
                        obs_n = con.normalizer.normalize(obs) if config.dqn.normalize else obs
                        if config.dqn.action_masking:
                            action = con.agent(obs_n, action_mask)
                        else:
                            action = con.agent(obs_n)
                next_obs, reward, terminated, truncated, info = env.step(action)
                terminated = terminated or truncated  # Envs that run into a timeout also terminate
                next_obs = tf_obs_callback(next_obs)
                observations.append(next_obs)
                actions.append(action)
                rewards.append(reward)
                infos.append(info)
                total_reward += reward
                if len(rewards) == config.dqn.multistep:
                    sum_r = sum([rewards[i] * config.gamma**i for i in range(config.dqn.multistep)])
                    sample = serializer.serialize_sample({
                        "obs": observations[0],
                        "action": actions[0],
                        "reward": sum_r,
                        "nextObs": observations[-1],
                        "done": terminated,
                        "info": infos[-1],
                        "modelId": model_id
                    })
                    con.push_sample(sample)
                    if sample_gauge:
                        current_gauge_cnt += 1
                        tnow = time.time()
                        if tnow - current_gauge_start_time > 60:
                            td = tnow - current_gauge_start_time
                            sample_gauge.value = int(current_gauge_cnt * 60 / td)
                            current_gauge_cnt = 0
                            current_gauge_start_time = tnow
                obs = next_obs
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
                    sample = serializer.serialize_sample({
                        "obs": observations[i],
                        "action": actions[i],
                        "reward": sum_r,
                        "nextObs": observations[-1],
                        "done": terminated,
                        "info": infos[-1],
                        "modelId": model_id
                    })
                    con.push_sample(sample)
                    if sample_gauge:
                        current_gauge_cnt += 1
                        tnow = time.time()
                        if tnow - current_gauge_start_time > 60:
                            td = tnow - current_gauge_start_time
                            sample_gauge.value = int(current_gauge_cnt * 60 / td)
                            current_gauge_cnt = 0
                            current_gauge_start_time = tnow
                noise.reset()
                tel = serializer.serialize_telemetry({
                    "reward": total_reward,
                    "steps": steps,
                    "obs": obs,
                    "eps": eps
                })
                con.push_telemetry(tel)
            if episode_end_callback is not None:
                episode_end_callback()
        logger.info("Exiting training")
    finally:
        env.close()
        con.close()
