"""The PPO client module contains the sampling loop for PPO on worker nodes.

Note:
    The implementation of PPO is synchronous. Therefore, the client and server are currently not
    resilient against disconnects.
"""
import logging
import time
from multiprocessing import Event
from types import SimpleNamespace
from typing import Callable

import gymnasium as gym

from soulsai.distributed.client.connector import PPOConnector

logger = logging.getLogger(__name__)


def ppo_client(config: SimpleNamespace,
               tf_obs_callback: Callable,
               encode_sample: Callable,
               encode_tel: Callable,
               episode_end_callback: Callable | None = None):
    """PPO client main function.

    Data processing, sample encodings etc. are configurable to customize the client for different
    environments. Messages to the redis client have to be encoded since redis messages can't use
    arbitrary data types.

    Args:
        config: The training configuration.
        tf_obs_callback: Callback to transform environment observations into agent inputs.
        encode_sample: Function to encode sample messages for redis.
        encode_tel: Function to encode the telemetry information for redis.
        episode_end_callback: Callback for functions that should be called at the end of an episode.
    """
    logging.basicConfig(level=config.loglevel)
    logging.getLogger("soulsai").setLevel(config.loglevel)
    logger.info("Launching PPO client")

    stop_flag = Event()
    if config.enable_interrupt:
        import keyboard  # Keyboard should not be imported in Docker during testing

        def exit_callback():
            stop_flag.set()

        keyboard.add_hotkey("enter", exit_callback)
        logger.info("Press 'Enter' to end training")

    env = gym.make(config.env)
    con = PPOConnector(config)
    con.sync()  # Wait for the new model to download

    logger.info("Client node running")
    try:
        episode_id = 0
        ppo_steps = 0
        while not stop_flag.is_set() and episode_id != config.max_episodes:
            episode_id += 1
            obs = tf_obs_callback(env.reset()[0])
            terminated = False
            total_reward = 0.
            steps = 1
            while not terminated and not stop_flag.is_set():
                action, prob = con.agent.get_action(obs)
                next_obs, reward, next_terminated, truncated, _ = env.step(action)
                next_terminated = next_terminated or truncated
                next_obs = tf_obs_callback(next_obs)
                total_reward += reward
                con.push_sample(con.agent.model_id, ppo_steps,
                                encode_sample(obs, action, prob, reward, terminated))
                logger.debug(f"Pushed sample {ppo_steps} for model {con.agent.model_id}")
                obs = next_obs
                terminated = next_terminated
                steps += 1
                ppo_steps += 1
                if config.step_delay:
                    time.sleep(config.step_delay)
                if ppo_steps == config.ppo.n_steps:
                    sample = encode_sample(next_obs, 0, [0], 0, terminated)
                    con.push_sample(con.agent.model_id, ppo_steps, sample)
                    ppo_steps = 0
                    con.sync(config.ppo.client_sync_timeout)  # Wait for the new model
            con.push_telemetry(encode_tel(total_reward, steps, obs))
            if episode_end_callback is not None:
                episode_end_callback()
            if con.shutdown.is_set():
                break
    finally:
        env.close()
        con.close()
