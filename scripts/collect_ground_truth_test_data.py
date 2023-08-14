from pathlib import Path

import gymnasium
import pandas as pd
import soulsgym  # noqa: F401


def main():
    env = gymnasium.make("SoulsGymIudex-v0")
    n_episodes = 10
    observations, actions, rewards, infos, episodes, steps = [], [], [], [], [], []
    for episode in range(n_episodes):
        done, step = False, 0
        obs, info = env.reset()
        observations.append(obs)
        infos.append(info)
        episodes.append(episode)
        steps.append(step)
        while not done:
            step += 1
            action = env.action_space.sample()
            obs, reward, truncated, terminated, info = env.step(action)
            done = terminated or truncated
            observations.append(obs)
            actions.append(action)
            rewards.append(reward)
            infos.append(info)
            episodes.append(episode)
            steps.append(step)

    path = Path(__file__).parents[1] / "test" / "pytest" / "data" / "iudex_env_dummy_data.csv"
    data = pd.DataFrame({
        "observations": observations,
        "actions": actions,
        "rewards": rewards,
        "infos": infos,
        "episodes": episodes,
        "steps": steps
    })
    data.head()
    data.to_csv(path, index=False)


if __name__ == "__main__":
    main()
