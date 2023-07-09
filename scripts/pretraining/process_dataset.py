from pathlib import Path
import argparse
import json
import multiprocessing as mp

import pandas as pd
from tqdm import tqdm
import torch
from torchvision.io import read_image

from soulsai.data.transformation import GameStateTransformer


def create_train_test_split(test_fraction):
    root_path = Path(__file__).parents[2] / "data" / "soulsgym_dataset"
    folders = [p for p in root_path.iterdir() if p.is_dir()]
    n_test_folders = int(len(folders) * test_fraction)
    train_folders = folders[:-n_test_folders]
    test_folders = folders[-n_test_folders:]
    train_path = root_path / "train_data"
    train_path.mkdir()
    for idx, folder in enumerate(train_folders):
        folder.rename(train_path / f"{idx:04d}")
    test_path = root_path / "test_data"
    test_path.mkdir()
    for idx, folder in enumerate(test_folders):  # Rename so that the folders are numbered from 0
        folder.rename(test_path / f"{idx:04d}")


def create_annotations(path):
    data_folders = [p for p in path.iterdir() if p.is_dir()]
    files = [f for d in data_folders for f in d.iterdir() if f.suffix == ".png"]
    local_idx = []
    folders = []
    pbar = tqdm(total=len(files), desc="Creating annotations")
    for folder in data_folders:
        for img in [i for i in folder.iterdir() if i.suffix == ".png"]:
            local_idx.append(img.stem)
            folders.append(folder.stem)
            pbar.update(1)
    train_df = pd.DataFrame({"local_idx": local_idx, "folder": folders})
    train_df.to_csv(path / "annotations.csv")


def convert_observations(path):
    obs_transformer = GameStateTransformer()
    annotations = pd.read_csv(path / "annotations.csv")
    all_obs = torch.empty((len(annotations), 74), dtype=torch.float32)
    pbar = tqdm(total=len(annotations), desc="Creating observations")
    labels = []
    for idx, row in annotations.iterrows():
        folder, local_idx = row["folder"], row["local_idx"]
        if local_idx == 0:
            obs_transformer.reset()
        with open(path / f"{folder:04d}" / f"{local_idx:04d}.json", "r") as f:
            obs = json.load(f)
        labels.append(obs)
        obs = torch.tensor(obs_transformer.transform(obs), dtype=torch.float32)
        torch.save(obs, path / f"{folder:04d}" / f"{local_idx:04d}_obs.pt")
        all_obs[idx] = obs
        pbar.update(1)
    torch.save(all_obs, path / "all_observations.pt")
    pd.DataFrame(labels).to_csv(path / "labels.csv", index=False)


def _load_img_data(path):
    imgs = [read_image(str(img_path)) for img_path in path.iterdir() if img_path.suffix == ".png"]
    return int(path.stem), torch.stack(imgs)  # ID, tensor of images


def create_single_img_file(path):
    paths = [p for p in path.iterdir() if p.is_dir()]
    len_episodes = [len([i for i in p.iterdir() if i.suffix == ".png"]) for p in paths]
    pool = mp.Pool(max(mp.cpu_count() - 1, 1))
    img_tensor = torch.zeros((sum(len_episodes), 3, 90, 160), dtype=torch.uint8)
    for x in tqdm(pool.imap(_load_img_data, paths), total=len(paths)):
        low = sum(len_episodes[:x[0]])
        high = sum(len_episodes[:x[0] + 1]) if x[0] < len(len_episodes) - 1 else None
        img_tensor[low:high] = x[1]
    torch.save(img_tensor, path / "all_images.pt")
    # We have to iterate so we don't run out of memory
    mean, sq_std = 0, 0
    for img in tqdm(img_tensor, "Calculating mean and std for images"):
        mean += img.float().mean(dim=(1, 2))
        sq_std += img.float().std(dim=(1, 2))**2
    mean /= len(img_tensor)
    std = torch.sqrt(sq_std / len(img_tensor))  # std = sqrt((std1^2 + std2^2 + ... + stdN^2) / N)
    torch.save((mean, std), path / "mean_std.pt")


def main(args):
    create_train_test_split(args.test_fraction)
    train_path = Path(__file__).parents[2] / "data" / "soulsgym_dataset" / "train_data"
    test_path = Path(__file__).parents[2] / "data" / "soulsgym_dataset" / "test_data"
    print("Creating train annotations...")
    create_annotations(train_path)
    print("Creating test annotations...")
    create_annotations(test_path)
    print("Creating train observations...")
    convert_observations(train_path)
    print("Creating test observations...")
    convert_observations(test_path)
    print("Creating single file for all train images...")
    create_single_img_file(train_path)
    print("Creating single file for all test images...")
    create_single_img_file(test_path)


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--test_fraction", type=float, required=True)
    args = argparser.parse_args()
    assert 0. < args.test_fraction < 1.
    main(args)
