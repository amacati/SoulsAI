"""The PPO client module contains the sampling loop for PPO on worker nodes.

Note:
    The implementation of PPO is synchronous. Therefore, the client and server are currently not
    resilient against disconnects.
"""
from __future__ import annotations

import logging
import time
from multiprocessing import Event
from typing import TYPE_CHECKING

import torch
import gymnasium

import soulsai.wrappers
from soulsai.wrappers import TensorDictWrapper
from soulsai.utils import namespace2dict
from soulsai.distributed.common.serialization import serialize
from soulsai.distributed.client.connector import PPOConnector

if TYPE_CHECKING:
    from types import SimpleNamespace

logger = logging.getLogger(__name__)


def ppo_client(config: SimpleNamespace):
    """PPO client main function.

    Data processing, sample encodings etc. are configurable to customize the client for different
    environments. Messages to the redis client have to be encoded since redis messages can't use
    arbitrary data types.

    Args:
        config: The training configuration.
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

    env = gymnasium.make(config.env.name)
    for wrapper, wrapper_args in namespace2dict(config.env.wrappers).items():
        env = getattr(soulsai.wrappers, wrapper)(env, **(wrapper_args["kwargs"] or {}))
    env = TensorDictWrapper(env)  # Convert all outputs to TensorDicts

    con = PPOConnector(config)
    con.sync()  # Wait for the new model to download

    logger.info("Client node running")
    try:
        episode_id = 0
        ppo_steps = 0
        while not stop_flag.is_set() and episode_id != config.max_episodes:
            episode_id += 1
            steps, episode_reward = 1, 0.
            sample = env.reset()
            obs = sample["obs"]
            done = False

            while not done and not stop_flag.is_set():
                action, prob = con.agent.get_action(obs)
                sample = env.step(action)
                sample["obs"] = obs
                episode_reward += sample["reward"]
                sample["prob"] = prob
                sample["model_id"] = torch.tensor([con.model_id] * env.num_envs)
                sample["client_id"] = torch.tensor([con.client_id] * env.num_envs)
                sample["step_id"] = torch.tensor([ppo_steps] * env.num_envs)

                con.push_sample(serialize(sample))
                logger.debug(f"Pushed sample {ppo_steps} for model {con.agent.model_id}")
                obs = sample["next_obs"]
                done = torch.all(sample["terminated"] | sample["truncated"])
                steps += 1
                ppo_steps += 1

                if config.step_delay:
                    time.sleep(config.step_delay)
                if ppo_steps == config.ppo.n_steps:
                    sample["obs"] = sample["next_obs"]
                    sample["action"] = [0] * env.num_envs
                    sample["prob"] = [0] * env.num_envs
                    sample["reward"] = [0] * env.num_envs
                    sample["step_id"] = [ppo_steps] * env.num_envs
                    con.push_sample(serialize(sample))
                    ppo_steps = 0
                    con.sync(config.ppo.client_sync_timeout)  # Wait for the new model
            episode_info = {"ep_reward": episode_reward, "ep_steps": steps, "obs": obs, "eps": 0}
            con.push_episode_info(serialize(episode_info))
            if con.shutdown.is_set():
                break
    finally:
        env.close()
        con.close()
