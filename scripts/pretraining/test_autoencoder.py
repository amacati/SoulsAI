from pathlib import Path

import torch
from torchvision import transforms
import matplotlib.pyplot as plt
import einops
from soulsai.data.dataset import SoulsGymImageDataset
from soulsai.core.models import AutoEncoder


def load_data(in_memory: bool = False, device: str = "cpu"):
    data_path = Path(__file__).parents[2] / "data" / "soulsgym_dataset" / "train_data"
    mean, std = torch.load(data_path / "mean_std.pt")
    img_transform = transforms.Compose(
        [transforms.ConvertImageDtype(torch.float),
         transforms.Normalize(mean=mean, std=std)])
    dataset = SoulsGymImageDataset(data_path,
                                   in_memory=in_memory,
                                   device=device,
                                   transform=img_transform)
    return dataset, mean, std


def compare_images(x, y, y_diff):
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    ax[0].imshow(x)
    ax[1].imshow(y)
    ax[2].imshow(y_diff)
    plt.show()


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # data, mean, std = load_data(in_memory=False, device=device)

    model = AutoEncoder()
    encoder_dict = torch.load(
        Path(__file__).parents[2] / "saves/autoencoder/3/autoencoder_checkpoint.pt")
    model.load_state_dict(encoder_dict)
    model.to(device).eval()

    x = torch.ones((3, 90, 160))
    x[..., 30:60, :] = 0
    x[..., 60:100] = 0
    x_img = einops.rearrange(x, "c h w -> h w c")

    with torch.no_grad():
        y_old = torch.zeros((90, 160, 3))
        while True:
            # x = random.choice(data)
            x = x.to(device)
            y = model(x.unsqueeze(0))[0]
            x, y = x.cpu(), y.cpu()
            y = einops.rearrange(y, "c h w -> h w c")
            compare_images(x_img, y, (y - y_old) / 2)
            y_old = y


if __name__ == "__main__":
    main()
