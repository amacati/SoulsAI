from pathlib import Path
import json
from types import SimpleNamespace

import numpy as np
import torch
import gymnasium
import fire
import soulsgym

import soulsai.wrappers
from soulsai.utils import namespace2dict, dict2namespace
from soulsai.core.agent import DistributionalDQNClientAgent, DQNClientAgent, ClientAgent
from soulsai.core.normalizer import AbstractNormalizer, normalizer_cls


def load_config(path: Path) -> SimpleNamespace:
    with open(path, "r") as f:
        return dict2namespace(json.load(f))


def load_agent(path: Path, config: SimpleNamespace) -> ClientAgent:
    if config.dqn.variant == "distributional":
        agent_cls = DistributionalDQNClientAgent
    else:
        agent_cls = DQNClientAgent
    agent = agent_cls(config.dqn.network.type, namespace2dict(config.dqn.network.kwargs),
                      config.device)
    agent.load(path)
    return agent


def evaluate_iudex(n_evals: int = 10):
    # Load config and agent
    path = Path(__file__).parents[1] / "saves/eval/iudex"
    config_phase_1 = load_config(path / "phase1/config.json")
    config_phase_2 = load_config(path / "phase2/config.json")
    agent_phase_1 = load_agent(path / "phase1/best_model/agent.pt", config_phase_1)
    agent_phase_2 = load_agent(path / "phase2/best_model/agent.pt", config_phase_2)
    raise NotImplementedError("Transforms are not implemented yet.")
    config = config_phase_1

    # Create environment
    env = gymnasium.make("SoulsGymIudexDemo-v0")
    for wrapper, wrapper_args in namespace2dict(config.env.wrappers).items():
        env = getattr(soulsai.wrappers, wrapper)(env, **(wrapper_args["kwargs"] or {}))

    # Create stats
    episode_rewards, episode_steps, wins = [], [], []

    for _ in range(n_evals):
        obs, info = env.reset()
        action_mask = np.zeros(config.env.n_actions) if config.dqn.action_masking else None
        if action_mask is not None:
            action_mask[info["allowed_actions"]] = 1
        terminated, truncated = False, False
        steps, episode_reward = 1, 0.
        agent = agent_phase_1
        while not terminated or truncated:
            obs_n = obs
            # Convert numpy or torch tensor to float32
            obs_n = obs_n.astype(np.float32) if isinstance(obs_n, np.ndarray) else obs_n.float()
            action = agent(obs_n, action_mask) if action_mask is not None else agent(obs_n)
            obs, reward, terminated, truncated, info = env.step(action)
            if action_mask is not None:
                action_mask[:] = 0
                action_mask[info["allowed_actions"]] = 1
            episode_reward += reward
            steps += 1
            if env.phase == 2:
                agent = agent_phase_2
        episode_rewards.append(episode_reward)
        episode_steps.append(steps)
        wins.append(info["boss_hp"] <= 0)
        # Save every episode to prevent data loss
        with open(path / "results.json", "w") as f:
            json.dump({"rewards": episode_rewards, "steps": episode_steps, "wins": wins}, f)

    print((f"Average reward: {np.mean(episode_rewards):.2f}, "
           f"average steps: {np.mean(episode_steps):.0f}, "
           f"win rate: {np.mean(wins):.4f}"))


if __name__ == "__main__":
    fire.Fire(evaluate_iudex)
