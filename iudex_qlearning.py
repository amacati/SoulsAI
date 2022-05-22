from pathlib import Path

import gym
import soulsgym  # noqa: F401
from soulsgym.core.game_state import GameState
from tqdm import tqdm
import numpy as np
import json

from agent import DQNAgent
from replay_buffer import ExperienceReplayBuffer
from utils import running_mean, fill_buffer, gamestate2np
from visualization import save_plots
from scheduler import EpsilonScheduler


if __name__ == "__main__":
    # Create environment
    env = gym.make("SoulsGymIudex-v0")
    # RL parameters
    gamma = 0.99
    n_episodes = 5000
    n_actions = env.action_space.n

    gamestate = GameState()
    gamestate.player_max_hp = 1
    gamestate.player_max_sp = 1
    gamestate.boss_max_hp = 1
    n_states = len(gamestate2np(gamestate))

    # Learning parameters
    lr = 1e-3
    eps_max = [0.99, 0.1, 0.1]
    eps_min = [0.1, 0.1, 0.01]
    eps_steps = [1500, 1500, 1500]
    grad_clip = 1.5
    q_clip = 200.
    buffer_size = 100
    batch_size = 64

    episodes_reward = []  # Contains the total reward per episode
    episodes_steps = []   # Contains the number of steps per episode
    iudex_hp = []
    wins = []

    agent = DQNAgent(n_states, n_actions, lr, gamma, grad_clip, q_clip)
    buffer = ExperienceReplayBuffer(maximum_length=buffer_size)
    eps_scheduler = EpsilonScheduler(eps_max, eps_min, eps_steps, zero_ending=True)
    path = Path(__file__).parent / "replay_buffer.pkl"
    fill_buffer(buffer, env, buffer_size, load=True, save=True, path=path)

    status_bar = tqdm(total=n_episodes, desc="Episodes: ", position=0, leave=False)
    reward_log = tqdm(total=0, position=1, bar_format='{desc}', leave=False)

    try:
        for i in range(n_episodes):
            done = False
            state = env.reset()
            ep_reward = 0.
            t = 0
            eps = eps_scheduler.epsilon
            eps_scheduler.step()
            while not done:
                state_A = gamestate2np(state)
                action = env.action_space.sample() if np.random.rand() < eps else agent(state_A)
                next_state, reward, done, _ = env.step(action)
                buffer.append((state, action, reward, next_state, done))
                ep_reward += reward
                state = next_state
                t += 1

                if len(buffer) > batch_size:
                    states, actions, rewards, next_states, dones = buffer.sample_batch(batch_size)
                    states = np.array([gamestate2np(state) for state in states])
                    next_states = np.array([gamestate2np(next_state) for next_state in next_states])
                    actions, rewards, dones = map(np.array, (actions, rewards, dones))
                    agent.train(states, actions, rewards, next_states, dones)

            episodes_reward.append(ep_reward)
            episodes_steps.append(t)
            iudex_hp.append(state.boss_hp)
            wins.append(int(iudex_hp == 0))

            if i % 100 == 0:
                agent.save(Path(__file__).parent)
                fig_path = Path(__file__).parent / "training_progress.png"
                save_plots(episodes_reward, episodes_steps, iudex_hp, wins, fig_path)
                stats_path = Path(__file__).parent / "stats.json"
                with open(stats_path, "w") as f:
                    json.dump({"rewards": episodes_reward, "steps": episodes_steps,
                            "iudex_hp": iudex_hp, "wins": wins}, f)

            desc = "Current average reward: {:.2f}".format(running_mean(episodes_reward, 50)[-1])
            reward_log.set_description_str(desc)
            status_bar.update()

            if i > 50 and running_mean(wins, 50)[-1] > 0.5:
                print("Training goal reached!")
                break
    finally:
        env.close()
