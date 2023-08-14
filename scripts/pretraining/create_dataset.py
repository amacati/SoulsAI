import gymnasium
import soulsgym  # noqa: F401
from PIL import Image
from pathlib import Path
import json
from tqdm import tqdm


def gs_to_json(gs):
    for key in ("player_hp", "player_sp", "boss_hp", "player_animation_duration",
                "boss_animation_duration"):
        gs[key] = float(gs[key])
    for key in ("player_pose", "boss_pose", "camera_pose"):
        gs[key] = gs[key].tolist()
    return gs


def main():
    env = gymnasium.make("SoulsGymIudex-v0", game_speed=3., init_pose_randomization=True)
    try:
        ep_id = 0
        n_samples = 100_000
        samples = 0
        root = Path(__file__).parents[2] / "data" / "soulsgym_dataset"
        pbar = tqdm(total=n_samples, desc="Creating dataset")
        while samples < n_samples:
            obs, info = env.reset()
            done = False
            ep_dir = root / str(ep_id)
            ep_dir.mkdir(parents=True, exist_ok=False)
            images, game_states = [], []
            images.append(Image.fromarray(obs["img"]))
            del obs["img"]
            obs["reward"] = 0.
            obs["info"] = info
            game_states.append(obs)
            while not done:
                action = env.action_space.sample()
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                images.append(Image.fromarray(obs["img"]))
                del obs["img"]
                obs["reward"] = reward
                obs["info"] = info
                game_states.append(obs)
            for i, (img, gs) in enumerate(zip(images, game_states)):
                img.save(ep_dir / f"{i:04d}.png")
                with open(ep_dir / f"{i:04d}.json", "w") as f:
                    json.dump(gs_to_json(gs), f)
            samples += len(game_states)
            pbar.update(len(game_states))
            ep_id += 1
        pbar.close()
    finally:
        env.close()


if __name__ == "__main__":
    main()
