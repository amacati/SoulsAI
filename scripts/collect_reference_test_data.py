from pathlib import Path
import json

import numpy as np
import gymnasium
import soulsgym  # noqa: F401

from soulsai.data.transformation import GameStateTransformer


def maybe_numpy_to_list(x):
    if isinstance(x, np.ndarray):
        return x.tolist()
    return x


def main():
    env = gymnasium.make("SoulsGymIudex-v0", game_speed=3., init_pose_randomization=True)
    transformer = GameStateTransformer()
    try:
        n_episodes = 2
        observations, actions, rewards, dones, infos, episodes, steps = [], [], [], [], [], [], []
        gt_observations = []
        for episode in range(n_episodes):
            done, step = False, 0
            obs, info = env.reset()
            observations.append(obs)
            gt_observations.append(transformer.transform(obs))
            infos.append(info)
            episodes.append(episode)
            steps.append(step)
            actions.append(0)
            rewards.append(0)
            dones.append(done)
            while not done:
                step += 1
                action = env.action_space.sample()
                obs, reward, truncated, terminated, info = env.step(action)
                done = terminated or truncated
                observations.append(obs)
                gt_observations.append(transformer.transform(obs))
                actions.append(action)
                rewards.append(reward)
                dones.append(done)
                infos.append(info)
                episodes.append(episode)
                steps.append(step)
            episode += 1
            transformer.reset()

        path = Path(__file__).parents[1] / "test/pytest/data/iudex_env_reference_input.json"
        data = {
            "observations": [{key: maybe_numpy_to_list(value)
                              for key, value in obs.items()}
                             for obs in observations],
            "actions": [int(action) for action in actions],
            "rewards": [float(reward) for reward in rewards],
            "dones": [bool(done) for done in dones],
            "infos": infos,
            "episodes": episodes,
            "steps": steps
        }
        with open(path, "w") as f:
            json.dump(data, f)
        path = path.parent / "iudex_env_reference_output.json"
        with open(path, "w") as f:
            json.dump([obs.tolist() for obs in gt_observations], f)
    finally:
        env.close()


if __name__ == "__main__":
    main()
