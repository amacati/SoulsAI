import torch
from torch.utils.data import Dataset
from torchvision.io import read_image
import pandas as pd
import einops


class SoulsGymImageDataset(Dataset):

    def __init__(self, root_dir, device, in_memory, input_transform=None, target_transform=None):
        self.root_dir = root_dir
        self.input_transform = input_transform
        self.target_transform = target_transform
        self.device = device
        self._in_memory = in_memory
        if in_memory:
            data_path = self.root_dir / "all_images.pt"
            assert data_path.exists(), "Dataset has not been preprocessed for in-memory loading."
            self.data = torch.load(self.root_dir / "all_images.pt")
        self.annotations = pd.read_csv(root_dir / "annotations.csv")

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx: int) -> torch.Tensor:
        data = self._getitem_from_memory(idx) if self._in_memory else self._getitem_from_disk(idx)
        if self.device:
            data = data.to(self.device)
        data_input = self.input_transform(data) if self.input_transform else data
        data_target = self.target_transform(data) if self.target_transform else data
        return data_input, data_target

    def _getitem_from_memory(self, idx: int) -> torch.Tensor:
        return self.data[idx]

    def _getitem_from_disk(self, idx):
        row = self.annotations.iloc[idx]
        folder, local_idx = row["folder"], row["local_idx"]
        return read_image(str(self.root_dir / f"{folder:04d}" / f"{local_idx:04d}.png"))


class SoulsGymDataset(Dataset):

    def __init__(self,
                 root_dir,
                 device=None,
                 in_memory=False,
                 transform=None,
                 target_transform=None):
        self.annotations = pd.read_csv(root_dir / "annotations.csv")
        self.root_dir = root_dir
        self.transform = transform
        self.target_transform = target_transform
        self.device = device
        self._in_memory = in_memory
        if in_memory:
            data_path = self.root_dir / "all_images.pt"
            assert data_path.exists(), "Dataset has not been preprocessed for in-memory loading."
            self.data = torch.load(self.root_dir / "all_images.pt")
            # The observations have the following entries:
            # 0-3: player_hp, player_sp, boss_hp, boss_distance
            # 4-15: player_pos, player_rot, boss_pos, boss_rot, camera_rot
            # 16-17: player_animation_duration, boss_animation_duration
            # 17-48: player_animation_onehot
            # 49-70: boss_animation_onehot
            self.observations = torch.load(self.root_dir / "all_observations.pt")
        self.labels = pd.read_csv(self.root_dir / "labels.csv")

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        if self._in_memory:
            data, obs = self._getitem_from_memory(idx)
        else:
            data, obs = self._getitem_from_disk(idx)
        if self.device:
            data = data.to(self.device)
            obs = obs.to(self.device)
        if self.transform:
            data = self.transform(data)
        if self.target_transform:
            obs = self.target_transform(obs)
        return data, obs

    def _getitem_from_memory(self, idx):
        # Compute the indices of the four images that form the observation. If the local index is
        # smaller than 3, we repeat the first image. The index is computed by taking the last four
        # indices and clamping them to the global index of the first image of the current episode.
        global_min_idx = max(0, idx - self.annotations["local_idx"].iloc[idx])
        img_idx = torch.maximum(torch.arange(idx - 3, idx + 1),
                                torch.ones(4, dtype=torch.int64) * global_min_idx)
        img = einops.rearrange(self.data[img_idx], "b c h w -> (b c) h w")
        return img, self.observations[idx]

    def _getitem_from_disk(self, idx):
        row = self.annotations.iloc[idx]
        folder, local_idx = row["folder"], row["local_idx"]
        folder_path = self.root_dir / f"{folder:04d}"
        obs = torch.load(self.root_dir / f"{folder:04d}" / f"{local_idx:04d}_obs.pt")
        img3 = read_image(str(self.root_dir / f"{folder:04d}" / f"{local_idx:04d}.png"))
        img0 = img1 = img2 = img3
        if local_idx == 1:
            img0 = img1 = img2 = read_image(str(folder_path / f"{local_idx-1:04d}.png"))
        elif local_idx == 2:
            img0 = img1 = read_image(str(folder_path / f"{local_idx-2:04d}.png"))
            img2 = read_image(str(folder_path / f"{local_idx-1:04d}.png"))
        elif local_idx >= 3:
            img0 = read_image(str(folder_path / f"{local_idx-3:04d}.png"))
            img1 = read_image(str(folder_path / f"{local_idx-2:04d}.png"))
            img2 = read_image(str(folder_path / f"{local_idx-1:04d}.png"))
        img = torch.cat((img0, img1, img2, img3), dim=0)
        return img, obs
