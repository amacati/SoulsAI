"""The DQN client module contains the sampling loop for DQN on worker nodes.

If specified, the training function is executed by a :class:`.ClientWatchdog` to restart the
sampling whenever the sampling rate drops below the expected value.
"""
import logging
from threading import Event
import time
from collections import deque
from types import SimpleNamespace
from multiprocessing.sharedctypes import Synchronized

import numpy as np
import gymnasium
import soulsai.wrappers
from soulsai.core.noise import get_noise_class, Noise
from soulsai.utils import namespace2dict
from soulsai.distributed.common.serialization import DQNSerializer
from soulsai.distributed.client.connector import DQNConnector
from soulsai.distributed.client.watchdog import ClientWatchdog, WatchdogGauge

logger = logging.getLogger(__name__)


def dqn_client(config: SimpleNamespace):
    """Wrap the the DQN client main function and automatically parameterize it.

    If the training is configured to use a watchdog, the watchdog starts the training and adds
    additional arguments to check the health of the training process by measuring the sampling rate.

    Args:
        config: The training configuration.
    """
    logging.basicConfig(level=config.loglevel)
    logging.getLogger("soulsai").setLevel(config.loglevel)
    logger.info("Launching DQN client")

    if config.watchdog.enable:
        minimum_samples_per_minute = config.watchdog.minimum_samples
        external_args = (config,)
        watchdog = ClientWatchdog(_dqn_client, minimum_samples_per_minute, external_args)
        watchdog.start()
    else:
        _dqn_client(config)


def _dqn_client(config: SimpleNamespace,
                stop_flag: Event = Event(),
                sample_gauge: Synchronized | None = None):
    """DQN client main function.

    Data processing, sample encodings etc. are configurable to customize the client for different
    environments. Messages to the redis client have to be encoded since redis messages can't use
    arbitrary data types.

    Args:
        config: The training configuration.
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
    # Create the environment and wrap it if wrappers are specified
    env = gymnasium.make(config.env.name, **namespace2dict(config.env.kwargs))
    for wrapper, wrapper_args in namespace2dict(config.env.wrappers).items():
        env = getattr(soulsai.wrappers, wrapper)(env, **(wrapper_args["kwargs"] or {}))
    # Create message serializer and action noise
    serializer = DQNSerializer(env_id=config.env.name)
    noise = get_noise_class(config.dqn.noise)(**namespace2dict(config.dqn.noise_kwargs))
    logger.info("Client node running")
    try:
        episode_id = 0
        observations = deque(maxlen=config.dqn.multistep + 1)
        actions = deque(maxlen=config.dqn.multistep)
        rewards = deque(maxlen=config.dqn.multistep)
        infos = deque(maxlen=config.dqn.multistep)
        gauge = WatchdogGauge(sample_gauge) if sample_gauge else None
        while (not stop_flag.is_set() and episode_id != config.max_episodes and  # noqa: W504
               not con.shutdown.is_set()):
            episode_id += 1
            obs, info = env.reset()
            action_mask = np.zeros(config.env.n_actions) if config.dqn.action_masking else None
            if action_mask:
                action_mask[info["allowed_actions"]] = 1
            terminated, truncated = False, False
            steps, episode_reward = 1, 0.
            observations.clear()
            actions.clear()
            rewards.clear()
            infos.clear()
            observations.append(obs)
            while not (terminated or truncated) and not stop_flag.is_set():
                with con:  # Context makes action and model_id consistent
                    action = _choose_action(con, obs, action_mask, noise)
                    eps, model_id = con.eps, con.model_id
                next_obs, reward, terminated, truncated, info = env.step(action)
                terminated = terminated or truncated  # Envs that run into a timeout also terminate
                observations.append(next_obs)
                actions.append(action)
                rewards.append(reward)
                infos.append(info)
                episode_reward += reward
                if len(rewards) == config.dqn.multistep:
                    sum_r = sum([rewards[i] * config.gamma**i for i in range(config.dqn.multistep)])
                    sample = serializer.serialize_sample({
                        "obs": observations[0],
                        "action": actions[0],
                        "reward": sum_r,
                        "nextObs": observations[-1],
                        "terminated": terminated,
                        "truncated": truncated,
                        "info": infos[-1],
                        "modelId": model_id
                    })
                    con.push_sample(sample)
                    if gauge:
                        gauge.inc(1)
                obs = next_obs
                if config.dqn.action_masking:
                    action_mask[:] = 0
                    action_mask[info["allowed_actions"]] = 1
                steps += 1
                if config.step_delay:  # Enable Dockerfiles to simulate slow clients
                    time.sleep(config.step_delay)
            # Sent the remaining samples for multistep > 1. We have no access to the remaining
            # required samples for a multistep reward. There are two cases:
            # 1) If the environment has terminated, we can send the samples since the remaining
            # trace calculates the MC reward correctly. The training step will not add the
            # Q estimate of future states to it.
            # 2) The environment was truncated. We cannot send these samples. The environment has
            # not terminated, so the training step would add the Q estimate discounted by
            # gamma ** multistep to the reward. However, our multistep samples are missing terms in
            # the reward sum because we can't generate the future samples for the estimate.
            # Therefore, the samples have to be discarded to prevent false estimates of the reward.
            if not stop_flag.is_set() and not truncated:
                for i in range(1, len(rewards)):
                    sum_r = sum(
                        [rewards[i + j] * config.gamma**j for j in range(config.dqn.multistep - i)])
                    sample = serializer.serialize_sample({
                        "obs": observations[i],
                        "action": actions[i],
                        "reward": sum_r,
                        "nextObs": observations[-1],
                        "terminated": terminated,
                        "truncated": truncated,
                        "info": infos[-1],
                        "modelId": model_id
                    })
                    con.push_sample(sample)
                if gauge:
                    gauge.inc(len(rewards) - 1)
            if terminated or truncated:
                ep_info = serializer.serialize_episode_info({
                    "epReward": episode_reward,
                    "epSteps": steps,
                    "eps": eps,
                    "modelId": model_id,
                    "obs": observations[-1]
                })
                con.push_episode_info(ep_info)
                noise.reset()
        logger.info("Exiting training")
    finally:
        env.close()
        con.close()


def _choose_action(con: DQNConnector, obs: np.ndarray, action_mask: np.ndarray,
                   noise: Noise) -> int:
    if np.random.rand() < con.eps:
        action = noise.sample(action_mask) if action_mask is not None else noise.sample()
    else:
        obs_n = con.normalizer.normalize(obs) if con.normalizer else obs
        # Convert numpy or torch tensor to float32
        obs_n = obs_n.astype(np.float32) if isinstance(obs_n, np.ndarray) else obs_n.float()
        action = con.agent(obs_n, action_mask) if action_mask is not None else con.agent(obs_n)
    return action
