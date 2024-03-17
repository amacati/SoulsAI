"""The DQN client module contains the sampling loop for DQN on worker nodes.

If specified, the training function is executed by a :class:`.ClientWatchdog` to restart the
sampling whenever the sampling rate drops below the expected value.
"""

from __future__ import annotations

import logging
import time
from collections import deque
from threading import Event
from typing import TYPE_CHECKING

import gymnasium
import torch
from tensordict import TensorDict

import soulsai.wrappers
from soulsai.distributed.client.connector import DQNConnector
from soulsai.distributed.client.watchdog import ClientWatchdog, WatchdogGauge
from soulsai.distributed.common.serialization import serialize
from soulsai.utils import namespace2dict
from soulsai.wrappers import TensorDictWrapper

if TYPE_CHECKING:
    from multiprocessing.sharedctypes import Synchronized
    from types import SimpleNamespace

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


def _dqn_client(
    config: SimpleNamespace, stop_flag: Event = Event(), sample_gauge: Synchronized | None = None
):
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
    env_factory = gymnasium.make if not config.env.vectorize else gymnasium.make_vec
    env = env_factory(config.env.name, **namespace2dict(config.env.kwargs))
    for wrapper, wrapper_args in namespace2dict(config.env.wrappers).items():
        env = getattr(soulsai.wrappers, wrapper)(env, **(wrapper_args["kwargs"] or {}))
    env = TensorDictWrapper(env)  # Convert all outputs to TensorDicts

    logger.info("Client node running")
    try:
        episode_id = 0
        multistep = getattr(config.dqn.agent.kwargs, "multistep", 1)
        assert hasattr(config.dqn.agent.kwargs, "gamma"), "No discount factor specified"
        gamma = config.dqn.agent.kwargs.gamma
        samples = deque(maxlen=multistep)
        gauge = WatchdogGauge(sample_gauge) if sample_gauge else None
        while (
            not stop_flag.is_set()
            and episode_id != config.max_episodes  # noqa: W504
            and not con.shutdown.is_set()
        ):
            episode_id += 1
            steps, episode_reward = 1, 0.0
            samples.clear()
            sample: TensorDict = env.reset()
            done = False
            while not done and not stop_flag.is_set():
                with con:  # Context makes action and model_id consistent
                    sample = con.transforms["obs"](sample)
                    sample = con.transforms["value"](con.agent(sample))
                    sample = con.transforms["action"](sample)
                    model_id = con.model_id
                # The observation has been altered by the observation transform. However, we want to
                # send the original observation to the server. Therefore, we store the original obs,
                # transform the sample, choose the action and then overwrite it with the original
                sample = env.step(sample["action"])
                samples.append(sample.clone())
                episode_reward += sample["reward"]
                if len(samples) == config.dqn.agent.kwargs.multistep:
                    rewards = torch.hstack([s["reward"] for s in samples])
                    discount_rewards: torch.Tensor = rewards * gamma ** torch.arange(multistep)
                    sum_rewards = discount_rewards.sum(dim=-1, keepdim=True)
                    msg = _sample_msg(
                        samples, sum_rewards, index=0, model_id=model_id, batch_size=env.num_envs
                    )
                    con.push_sample(serialize(msg))
                    gauge and gauge.inc(1)
                sample["obs"] = sample["next_obs"]
                steps += 1
                config.step_delay and time.sleep(config.step_delay)  # Enable Docker to slow clients
                done = torch.all(sample["terminated"] | sample["truncated"])
            if stop_flag.is_set():  # If the training was interrupted, don't send the episode info
                break
            # Sent the remaining samples for multistep > 1. We have no access to the remaining
            # required samples for a multistep reward. There are two cases:
            #
            # 1) If the environment has terminated, we can send the samples since the remaining
            # trace calculates the MC reward correctly. The training step will not add the
            # Q estimate of future observations to it.
            #
            # 2) The environment was truncated. We cannot send these samples. The environment has
            # not terminated, so the training step would add the Q estimate discounted by
            # gamma ** multistep to the reward. However, our multistep samples are missing terms in
            # the reward sum because we can't generate the future samples for the estimate.
            # Therefore, the samples have to be discarded to prevent false estimates of the reward.
            if not torch.any(sample["truncated"]):
                for i in range(1, len(rewards)):
                    rewards = torch.hstack([samples[i + j]["reward"] for j in range(multistep - i)])
                    discount_rewards: torch.Tensor = rewards * gamma ** torch.arange(multistep - i)
                    sum_rewards = discount_rewards.sum(dim=-1, keepdim=True)
                    msg = _sample_msg(
                        samples, sum_rewards, index=i, model_id=model_id, batch_size=env.num_envs
                    )
                    con.push_sample(serialize(msg))
                gauge and gauge.inc(len(rewards) - 1)
            if torch.any(sample["terminated"] | sample["truncated"]):
                ep_info = TensorDict(
                    {
                        "ep_reward": torch.tensor([episode_reward]),
                        "ep_steps": torch.tensor([steps]),
                        "model_id": torch.tensor([model_id] * env.num_envs),
                        "obs": sample["obs"],
                    },
                    batch_size=1,
                )
                if sample.get("info", None) is not None:
                    ep_info["info"] = sample["info"]
                con.push_episode_info(serialize(ep_info))
        logger.info("Exiting training")
    finally:
        env.close()
        con.close()


def _sample_msg(
    samples: deque[TensorDict], reward: float, index: int, model_id: int, batch_size: int
) -> TensorDict:
    """Create a sample message for the DQN server.

    Args:
        samples: The samples to be sent to the server.
        reward: The reward to be sent to the server.
        index: The index of the first sample in the deque.
        model_id: The model id of the agent that generated the samples.
        batch_size: The batch size of the samples.
    """
    msg = TensorDict(
        {
            "obs": samples[index]["obs"],
            "action": samples[index]["action"],
            "reward": reward,
            "next_obs": samples[-1]["next_obs"],
            "terminated": samples[-1]["terminated"],
            "truncated": samples[-1]["truncated"],
            "model_id": torch.tensor([model_id] * batch_size),
        },
        batch_size=batch_size,
    )
    if samples[-1].get("info", None) is not None:
        msg["info"] = samples[-1]["info"]
    return msg
