from pathlib import Path
import json

import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from sacred import Experiment
from sacred.observers import FileStorageObserver
from tqdm import tqdm
import einops

from soulsai.data.dataset import SoulsGymImageDataset
from soulsai.core.models import AutoEncoder

ex = Experiment("AutoEncoder pretraining")


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
    n_train_samples = int(len(dataset) * 0.8)
    n_val_samples = len(dataset) - n_train_samples
    train_data, val_data = torch.utils.data.random_split(dataset, [n_train_samples, n_val_samples])
    return train_data, val_data


@torch.no_grad()
def evaluate(model, val_dataloader, device):
    model.eval()
    loss = 0
    for x in val_dataloader:
        x = x.to(device)
        x_reconstruct = model(x)
        loss += F.mse_loss(x_reconstruct, x, reduction="sum").item()
    model.train()
    return loss


@ex.config
def config():
    n_epochs = 100  # noqa: F841
    lr = 1e-3  # noqa: F841
    lr_scheduler_kwargs = {  # noqa: F841
        "factor": 0.5,
        "patience": 10,
    }
    batch_size = 64  # noqa: F841
    embedding_dim = 128  # noqa: F841


@ex.main
def main(n_epochs: int, lr: float, lr_scheduler_kwargs: dict, batch_size: int, embedding_dim: int,
         _run):
    writer = SummaryWriter(Path(__file__).parents[2] / "saves" / "autoencoder" / str(_run._id))

    in_memory = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_data, val_data = load_data(in_memory=in_memory, device=device)

    data_path = Path(__file__).parents[2] / "data" / "soulsgym_dataset" / "train_data"
    mean, std = torch.load(data_path / "mean_std.pt")
    img_transform = transforms.Compose(
        [transforms.ConvertImageDtype(torch.float),
         transforms.Normalize(mean=mean, std=std)])
    dataset = SoulsGymImageDataset(data_path,
                                   in_memory=False,
                                   device=device,
                                   transform=img_transform)

    x0 = einops.repeat(dataset[0], "c h w -> r c h w", r=64).to(device)
    x0[...] = 1
    x0[..., 30:60, :] = 0
    x0[..., 60:100] = 0

    save_path = Path(__file__).parents[2] / "saves" / "autoencoder" / str(_run._id)

    model = AutoEncoder().to(device)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, **lr_scheduler_kwargs)

    n_workers = 0 if in_memory else 10
    train_dataloader = DataLoader(train_data, batch_size, num_workers=n_workers, shuffle=True)
    val_dataloader = DataLoader(val_data, batch_size, num_workers=n_workers, shuffle=True)

    losses = []
    # best_loss = float("inf")
    mean, std = mean.view((1, 3, 1, 1)).to(device), std.view((1, 3, 1, 1)).to(device)

    pbar = tqdm(range(n_epochs))
    for i in range(n_epochs):
        epoch_loss = 0
        for x in range(len(train_dataloader)):
            # x = x.to(device)
            x = x0
            optimizer.zero_grad()
            x_reconstruct = model(x)
            loss = F.mse_loss(x_reconstruct, x, reduction="sum")
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        eval_loss = 0
        # eval_loss = evaluate(model, val_dataloader, device)
        epoch_loss /= (len(train_data) * 14400)
        eval_loss /= (len(val_data) * 14400)
        pbar.set_postfix({'Loss': epoch_loss})
        pbar.update(1)
        losses.append(epoch_loss)
        writer.add_scalar("Train/Loss", epoch_loss, i)
        writer.add_scalar("Val/Loss", eval_loss, i)
        writer.add_scalar("Train/Learning rate", optimizer.param_groups[0]["lr"], i)
        lr_scheduler.step(epoch_loss)
        # if eval_loss < best_loss:
        torch.save(model.state_dict(), save_path / "autoencoder_checkpoint.pt")
        # best_loss = eval_loss
    save_path.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), save_path / "autoencoder.pt")
    stats = {"losses": losses}
    with open(save_path / "stats.json", "w") as f:
        json.dump(stats, f)


if __name__ == "__main__":
    sacred_dir = Path(__file__).parents[2] / "saves" / "autoencoder"
    ex.observers.append(FileStorageObserver(sacred_dir))
    ex.run_commandline()
