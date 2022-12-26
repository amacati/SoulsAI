from pathlib import Path
import json
import logging

import numpy as np
import gym
import soulsgym
import torch

from soulsai.utils import dict2namespace, namespace2dict
from soulsai.core.agent import DQNClientAgent
from soulsai.core.normalizer import Normalizer
from soulsai.data.transformation import GameStateTransformer

logger = logging.getLogger(__name__)


def main():
    ntests = 10
    nwins = 0
    ep_hp, ep_steps = [], []
    # Load config
    root_path = Path(__file__).parents[1] / "saves" / "checkpoint"
    with open(root_path / "config.json", "r") as f:
        config = dict2namespace(json.load(f))
    # Initialize agent, normalizers and environment
    agent = DQNClientAgent(config.dqn.network_type,
                           namespace2dict(config.dqn.network_kwargs))
    agent.load(root_path)
    norm_kwargs = {}
    if config.dqn.normalizer_kwargs is not None:
        norm_kwargs = namespace2dict(config.dqn.normalizer_kwargs)
    
    if config.dqn.normalize:
        normalizer = Normalizer(config.n_states, **norm_kwargs)
        normalizer.load_state_dict(torch.load(root_path / "normalizer.pt"))
    state_transform = GameStateTransformer().transform
    env_kwargs = namespace2dict(config.env_kwargs) if config.env_kwargs is not None else {}
    env = gym.make(config.env, **env_kwargs)
    try:
        for i in range(ntests):
            done = False
            ep_steps.append(0)
            state = state_transform(env.reset())
            if config.dqn.action_masking:
                valid_actions = env.current_valid_actions()
                action_mask = np.zeros(config.n_actions)
                action_mask[valid_actions] = 1

            while not done:
                state = normalizer.normalize(state) if config.dqn.normalize else state
                action = agent(state, action_mask) if config.dqn.action_masking else agent(state)
                next_state, reward, done, info = env.step(action)
                state = state_transform(next_state)
                ep_steps[i] += 1
                if config.dqn.action_masking:
                    action_mask[:] = 0
                    action_mask[info["allowed_actions"]] = 1
            ep_hp.append(state[2]*1037)
            nwins += reward > 2

        logger.info(f"Average HP per run: {sum(ep_hp)/ntests:.0f}, best HP: {min(ep_hp):.0f}")
        logger.info(f"Average steps per run: {sum(ep_steps)/ntests:.0f}, top steps: {max(ep_steps)}")
        logger.info(f"Boss has been beaten {nwins} out of {ntests} times.")
    finally:
        env.close()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
