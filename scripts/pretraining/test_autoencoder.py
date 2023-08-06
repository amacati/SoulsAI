from pathlib import Path
import random

import torch
from torchvision import transforms
import matplotlib.pyplot as plt
import einops
from soulsai.data.dataset import SoulsGymImageDataset
from soulsai.core.models import AutoEncoder


def load_data(in_memory: bool = False, device: str = "cpu"):
    data_path = Path(__file__).parents[2] / "data" / "soulsgym_dataset" / "train_data"
    mean, std = torch.load(data_path / "mean_std.pt")
    input_transform = transforms.Compose(
        [transforms.ConvertImageDtype(torch.float),
         transforms.Normalize(mean=mean, std=std)])
    target_transform = transforms.ConvertImageDtype(torch.float)
    dataset = SoulsGymImageDataset(data_path,
                                   in_memory=in_memory,
                                   device=device,
                                   input_transform=input_transform,
                                   target_transform=target_transform)
    return dataset


def compare_images(x, y, y_diff):
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    ax[0].imshow(x)
    ax[1].imshow(y)
    ax[2].imshow(y_diff)
    plt.show()


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = load_data(in_memory=False, device=device)

    model = AutoEncoder(2048)
    encoder_dict = torch.load(
        Path(__file__).parents[2] / "saves/autoencoder/8/checkpoints/best_autoencoder.pt")
    model.load_state_dict(encoder_dict)
    model.to(device).eval()

    with torch.no_grad():
        y_old = torch.zeros((90, 160, 3))
        while True:
            x_normed, x_original = random.choice(data)
            x_normed = x_normed.to(device)
            y = model(x_normed.unsqueeze(0))[0]
            y = einops.rearrange(y.cpu(), "c h w -> h w c")
            x_original = einops.rearrange(x_original.cpu(), "c h w -> h w c")
            compare_images(x_original, y, (y - y_old) / 2)
            y_old = y


if __name__ == "__main__":
    main()
