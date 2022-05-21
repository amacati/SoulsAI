from pathlib import Path
import random

import json
from uuid import uuid4
from tqdm import tqdm
import gym
import matplotlib.pyplot as plt

from ppo_agent import PPOAgent
from replay_buffer import EpisodeBuffer
from utils import running_average


if __name__ == "__main__":
    e_id = str(uuid4())

    for j in range(5):
        # General setup
        show_plot = False

        # Create environment
        env = gym.make("LunarLander-v2")

        # RL parameters
        gamma = 0.99
        n_episodes = 3000
        n_actions = env.action_space.n
        n_states = len(env.observation_space.low)

        # Learning parameters
        lr_actor = 5e-5  # 5e-5 best
        lr_critic = 1e-3  # 1e-3 best
        epochs = 40
        _lambda = 0.95
        epsilon = 0.2

        episodes_reward = []  # Contains the total reward per episode
        episodes_steps = []   # Contains the number of steps per episode

        agent = PPOAgent(n_states, n_actions, epochs, gamma, _lambda, epsilon, lr_actor, lr_critic)
        buffer = EpisodeBuffer()

        status_bar = tqdm(total=n_episodes, desc="Episodes: ", position=0, leave=False)
        reward_log = tqdm(total=0, position=1, bar_format='{desc}', leave=False)
        for i in range(n_episodes):
            done = False
            state = env.reset()
            ep_reward = 0.
            t = 0
            while not done:
                p_a = agent(state)
                action = random.choices(range(n_actions), p_a)[0]
                next_state, reward, done, _ = env.step(action)
                buffer.append((state, action, reward))
                ep_reward += reward
                state = next_state
                t += 1

            # Train the agent
            states, actions, rewards = buffer.sample()
            agent.train(states, actions, rewards)
            buffer.clear()

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
