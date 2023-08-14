"""Script to demo a full agent trained to defeat Iudex Gundyr.
"""
from pathlib import Path
import json
import logging
import time

import numpy as np
import gymnasium as gym
import soulsgym  # noqa: F401
import torch

from soulsai.utils import dict2namespace, namespace2dict
from soulsai.core.agent import DQNClientAgent
from soulsai.core.normalizer import get_normalizer_class
from soulsai.data.transformation import GameStateTransformer

logger = logging.getLogger(__name__)


def load_agent(path):
    with open(path / "config.json", "r") as f:
        config = dict2namespace(json.load(f))
    # Initialize agent, normalizers and environment
    agent = DQNClientAgent(config.dqn.network_type, namespace2dict(config.dqn.network_kwargs))
    agent.load(path)
    normalizer = None
    if config.dqn.normalizer:
        normalizer_cls = get_normalizer_class(config.dqn.normalizer)
        norm_kwargs = namespace2dict(config.dqn.normalizer_kwargs)
        normalizer = normalizer_cls(config.env.state_shape, **norm_kwargs)
        normalizer.load_state_dict(torch.load(path / "normalizer.pt"))
    obs_transform = GameStateTransformer().transform
    return agent, normalizer, obs_transform


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    ntests = 10
    nwins = 0
    ep_hp, ep_steps = [], []
    # Load agents
    path = Path(__file__).parents[1] / "saves" / "iudex_phase_1" / "best_model"
    agent_1, normalizer_1, obs_transform = load_agent(path)
    path = Path(__file__).parents[1] / "saves" / "iudex_phase_2" / "best_model"
    agent_2, normalizer_2, _ = load_agent(path)
    with open(path / "config.json", "r") as f:
        config = dict2namespace(json.load(f))

    # Initialize tracking stats and save folder
    replay_dir = Path(__file__).parents[1] / "saves" / "replay_data"
    replay_dir.mkdir(exist_ok=True)
    replay_dict = {}

    env = gym.make("SoulsGymIudexDemo-v0", game_speed=3.0)
    try:
        i = 0
        while nwins < 1:
            replay_dict[i] = []
            terminated = False
            ep_steps.append(0)
            obs, info = env.reset()
            obs = obs_transform(obs)
            phase = 1
            if config.dqn.action_masking:
                action_mask = np.zeros(config.env.n_actions)
                action_mask[info["allowed_actions"]] = 1
            while not terminated:
                normalizer = normalizer_1 if phase == 1 else normalizer_2
                agent = agent_1 if phase == 1 else agent_2
                obs = normalizer.normalize(obs) if config.dqn.normalizer else obs
                replay_dict[i].append({
                    "obs": obs.tolist(),
                    "action_mask": list(action_mask),
                    "timestamp": time.time()
                })
                action = agent(obs, action_mask) if config.dqn.action_masking else agent(obs)
                next_obs, reward, terminated, truncated, info = env.step(action)
                phase = next_obs["phase"]
                obs = obs_transform(next_obs)
                ep_steps[i] += 1
                if config.dqn.action_masking:
                    action_mask[:] = 0
                    action_mask[info["allowed_actions"]] = 1
            i += 1
            ep_hp.append(obs[2] * 1037)
            nwins += int(next_obs["boss_hp"] == 0)
            if next_obs["player_hp"] > 0:
                time.sleep(10)
        ntests = i
        logger.info(f"Average HP per run: {sum(ep_hp)/ntests:.0f}, best HP: {min(ep_hp):.0f}")
        logger.info((f"Average steps per run: {sum(ep_steps)/ntests:.0f}, "
                     f"top steps: {max(ep_steps)}"))
        logger.info((f"Boss has been beaten {nwins} out of {ntests} times ({nwins/ntests*100:.0f}%"
                     " winrate)."))
        # time.sleep(25)
    finally:
        env.close()
        print(replay_dict)
        with open(replay_dir / "replay_data.json", "w") as f:
            json.dump(replay_dict, f)
