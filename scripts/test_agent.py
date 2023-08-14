"""Script to test an agent trained on the SoulsGymIudex-v0 environment.

Load the agent in `saves/checkpoint`, run the environment for 10 episodes and report the stats.
"""
from pathlib import Path
import json
import logging

import numpy as np
import gymnasium as gym
import soulsgym  # noqa: F401
import torch

from soulsai.utils import dict2namespace, namespace2dict
from soulsai.core.agent import DQNClientAgent
from soulsai.core.normalizer import get_normalizer_class
from soulsai.data.transformation import GameStateTransformer

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    ntests = 10
    nwins = 0
    ep_hp, ep_steps = [], []
    # Load config
    root_path = Path(__file__).parents[1] / "saves" / "checkpoint"
    with open(root_path / "config.json", "r") as f:
        config = dict2namespace(json.load(f))
    # Initialize agent, normalizers and environment
    agent = DQNClientAgent(config.dqn.network_type, namespace2dict(config.dqn.network_kwargs))
    agent.load(root_path)
    normalizer = None
    if config.dqn.normalizer:
        normalizer_cls = get_normalizer_class(config.dqn.normalizer)
        norm_kwargs = namespace2dict(config.dqn.normalizer_kwargs)
        normalizer = normalizer_cls(config.env.state_shape, **norm_kwargs)
        normalizer.load_state_dict(torch.load(root_path / "normalizer.pt"))
    obs_transform = GameStateTransformer().transform
    env_kwargs = namespace2dict(config.env.kwargs) if config.env.kwargs else {}
    env = gym.make(config.env.name, **env_kwargs)
    try:
        for i in range(ntests):
            terminated = False
            ep_steps.append(0)
            obs, info = env.reset()
            obs = obs_transform(obs)
            if config.dqn.action_masking:
                action_mask = np.zeros(config.env.n_actions)
                action_mask[info["allowed_actions"]] = 1
            while not terminated:
                obs = normalizer.normalize(obs) if normalizer else obs
                action = agent(obs, action_mask) if config.dqn.action_masking else agent(obs)
                next_obs, reward, terminated, truncated, info = env.step(action)
                win = next_obs.boss_hp == 0
                obs = obs_transform(next_obs)
                ep_steps[i] += 1
                if config.dqn.action_masking:
                    action_mask[:] = 0
                    action_mask[info["allowed_actions"]] = 1
            ep_hp.append(obs[2] * 1037)
            nwins += win

        logger.info(f"Average HP per run: {sum(ep_hp)/ntests:.0f}, best HP: {min(ep_hp):.0f}")
        logger.info((f"Average steps per run: {sum(ep_steps)/ntests:.0f}, "
                     "top steps: {max(ep_steps)}"))
        logger.info((f"Boss has been beaten {nwins} out of {ntests} times ({nwins/ntests*100:.0f}%)"
                     "winrate)."))
    finally:
        env.close()
