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

from soulsai.data.dataset import SoulsGymImageDataset
from soulsai.core.models import AutoEncoder

ex = Experiment("AutoEncoder pretraining")


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
    n_train_samples = int(len(dataset) * 0.8)
    n_val_samples = len(dataset) - n_train_samples
    train_data, val_data = torch.utils.data.random_split(dataset, [n_train_samples, n_val_samples])
    return train_data, val_data


@torch.no_grad()
def evaluate(model, val_dataloader, device):
    model.eval()
    loss = 0
    for img_input, img_target in val_dataloader:
        img_input, img_target = img_input.to(device), img_target.to(device)
        x_reconstruct = model(img_input)
        loss += F.mse_loss(x_reconstruct, img_target, reduction="sum").item()
    model.train()
    return loss


@ex.config
def config():
    n_epochs = 100  # noqa: F841
    lr = 1e-2  # noqa: F841
    lr_scheduler_kwargs = {  # noqa: F841
        "factor": 0.5,
        "patience": 10,
    }
    batch_size = 128  # noqa: F841
    embedding_dim = 2048  # noqa: F841


@ex.main
def main(n_epochs: int, lr: float, lr_scheduler_kwargs: dict, batch_size: int, embedding_dim: int,
         _run):
    writer = SummaryWriter(Path(__file__).parents[2] / "saves" / "autoencoder" / str(_run._id))

    in_memory = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_data, val_data = load_data(in_memory=in_memory, device=device)

    model = AutoEncoder(embedding_dim).to(device)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, **lr_scheduler_kwargs)

    n_workers = 0 if in_memory else 10
    train_dataloader = DataLoader(train_data, batch_size, num_workers=n_workers, shuffle=True)
    val_dataloader = DataLoader(val_data, batch_size, num_workers=n_workers, shuffle=True)

    losses = []
    best_loss = float("inf")
    save_path = Path(__file__).parents[2] / "saves" / "autoencoder" / str(_run._id)
    (save_path / "checkpoints").mkdir(parents=True, exist_ok=True)

    pbar = tqdm(range(n_epochs))
    for i in pbar:
        epoch_loss = 0
        for img_input, img_target in train_dataloader:
            img_input, img_target = img_input.to(device), img_target.to(device)
            optimizer.zero_grad()
            x_reconstruct = model(img_input)
            loss = F.mse_loss(x_reconstruct, img_target, reduction="sum")
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        eval_loss = evaluate(model, val_dataloader, device)
        epoch_loss /= (len(train_data) * 160 * 90 * 3)
        eval_loss /= (len(val_data) * 160 * 90 * 3)
        pbar.set_postfix({'Loss': epoch_loss})
        losses.append(epoch_loss)
        writer.add_scalar("Train/Loss", epoch_loss, i)
        writer.add_scalar("Val/Loss", eval_loss, i)
        writer.add_scalar("Train/Learning rate", optimizer.param_groups[0]["lr"], i)
        lr_scheduler.step(epoch_loss)
        if eval_loss < best_loss:
            torch.save(model.state_dict(), save_path / "checkpoints/best_autoencoder.pt")
            best_loss = eval_loss
        if i % 10 == 0:
            torch.save(model.state_dict(), save_path / f"checkpoints/autoencoder_{i:03d}.pt")
    save_path.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), save_path / "autoencoder.pt")
    stats = {"losses": losses}
    with open(save_path / "stats.json", "w") as f:
        json.dump(stats, f)


if __name__ == "__main__":
    sacred_dir = Path(__file__).parents[2] / "saves" / "autoencoder"
    ex.observers.append(FileStorageObserver(sacred_dir))
    ex.run_commandline()
