from pathlib import Path

import json
from tqdm import tqdm
import numpy as np
import gym
import torch
import matplotlib.pyplot as plt
from uuid import uuid4

from soulsai.core.agent import DQNAgent
from soulsai.core.replay_buffer import ExperienceReplayBuffer, MultistepEpisodeBuffer, ImportanceReplayBuffer  # noqa
from soulsai.core.utils import running_average, fill_buffer


def state_tf(state: np.ndarray) -> np.ndarray:
    return np.array([*state[:4], np.sin(state[4]), np.cos(state[4]), *state[5:]])


if __name__ == "__main__":
    e_id = str(uuid4())

    for j in range(5):
        # General setup
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        show_plot = False

        # Create environment
        env = gym.make("LunarLander-v2")

        # RL parameters
        gamma = 0.99
        n_episodes = 400
        n_actions = env.action_space.n
        n_states = len(env.observation_space.low)

        # Learning parameters
        lr = 1e-3  # Slightly lower increases stability at the cost of improvement speed
        eps_max = 0.99
        eps_min = 0.01
        eps_decay = 0.9 * n_episodes
        grad_clip = 1.5
        buffer_size = 30000
        batch_size = 64
        train_epochs = 250
        multistep = 10

        episodes_reward = []  # Contains the total reward per episode
        episodes_steps = []   # Contains the number of steps per episode

        agent = DQNAgent(n_states+1, n_actions, lr, gamma, grad_clip)
        buffer = ExperienceReplayBuffer(maxlen=buffer_size)
        # buffer = ImportanceReplayBuffer(maxlen=buffer_size)
        fill_buffer(buffer, env, buffer_size, state_tf)

        status_bar = tqdm(total=n_episodes, desc="Episodes: ", position=0, leave=False)
        reward_log = tqdm(total=0, position=1, bar_format='{desc}', leave=False)
        for i in range(n_episodes):
            done = False
            state = state_tf(env.reset())
            ep_reward = 0.
            t = 0
            eps = max(eps_min, eps_max*(eps_min/eps_max)**(i/eps_decay))
            while not done:
                if np.random.rand() < eps:
                    action = env.action_space.sample()
                else:
                    action = agent(state)
                next_state, reward, done, _ = env.step(action)
                next_state = state_tf(next_state)
                buffer.append((state, action, reward, next_state, done))
                ep_reward += reward
                state = next_state
                t += 1

            # Train the Q functions
            for _ in range(train_epochs):
                states, actions, rewards, next_states, dones = buffer.sample_batch(batch_size)
                agent.train(states, actions, rewards, next_states, dones)

            episodes_reward.append(ep_reward)
            episodes_steps.append(t)
            desc = "Current average reward: {:.2f}".format(running_average(episodes_reward, 50)[-1])
            reward_log.set_description_str(desc)
            status_bar.update()

            if i > 50 and running_average(episodes_reward, 50)[-1] > 200:
                print("Training goal reached!")
                break

        env.close()

        path = Path(__file__).parent / "saves"
        agent.save(path)
        print("Model saved.")
        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(16, 9))
        ax[0].plot(range(len(episodes_reward)), episodes_reward, label='Episode reward')
        ax[0].plot(range(len(episodes_reward)), running_average(episodes_reward, 50),
                   label='Avg. episode reward')
        ax[0].set_xlabel('Episodes')
        ax[0].set_ylabel('Total reward')
        ax[0].set_title('Total Reward vs Episodes')
        ax[0].legend()
        ax[0].grid(alpha=0.3)
        ax[0].set_ylim([-400, 400])

        ax[1].plot(range(len(episodes_steps)), episodes_steps, label='Steps per episode')
        ax[1].plot(range(len(episodes_steps)), running_average(episodes_steps, 50),
                   label='Avg. number of steps per episode')
        ax[1].set_xlabel('Episodes')
        ax[1].set_ylabel('Total number of steps')
        ax[1].set_title('Total number of steps vs Episodes')
        ax[1].legend()
        ax[1].grid(alpha=0.3)
        if show_plot:
            plt.show()
        fig.savefig(Path(__file__).parent / ("stats" + str(j) + ".png"))

        savefile = Path(__file__).parent / "saves" / "results.json"

        if savefile.exists():
            with open(savefile, "r") as f:
                save = json.load(f)
            if save["id"] == e_id:
                save["episodes_rewards"].append(episodes_reward)
                save["episodes_steps"].append(episodes_steps)
            else:
                save = {"id": e_id, "episodes_rewards": [episodes_reward],
                        "episodes_steps": [episodes_steps]}
        else:
            save = {"id": e_id, "episodes_rewards": [episodes_reward],
                    "episodes_steps": [episodes_steps]}
        with open(savefile, "w") as f:
            json.dump(save, f)
