from pathlib import Path
import json

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from sacred import Experiment
from sacred.observers import FileStorageObserver
from tqdm import tqdm
from soulsgym.core.static import boss_animations
import einops

from soulsai.data.dataset import SoulsGymDataset
from soulsai.core.models import CNNEncoder

ex = Experiment("CNN pretraining")


def target_transform(obs):
    i = torch.argmax(obs[49:])
    assert i < 23
    return i


def load_data(in_memory: bool = False, device: str = "cpu"):
    data_path = Path(__file__).parents[2] / "data" / "soulsgym_dataset" / "train_data"
    mean, std = torch.load(data_path / "mean_std.pt")
    mean, std = einops.repeat(mean, "c -> (r c)", r=4), einops.repeat(std, "c -> (r c)", r=4)
    img_transform = transforms.Compose(
        [transforms.ConvertImageDtype(torch.float),
         transforms.Normalize(mean=mean, std=std)])

    dataset = SoulsGymDataset(data_path,
                              in_memory=in_memory,
                              device=device,
                              transform=img_transform,
                              target_transform=target_transform)

    n_train_samples = int(len(dataset) * 0.8)
    n_val_samples = len(dataset) - n_train_samples
    train_data, val_data = torch.utils.data.random_split(dataset, [n_train_samples, n_val_samples])
    return train_data, val_data


@torch.no_grad()
def evaluate(model, val_dataloader, device):
    model.eval()
    correct = 0
    for x, y in val_dataloader:
        x, y = x.to(device), y.to(device)
        y_pred = model(x)
        predicted = torch.argmax(y_pred, 1)
        correct += (predicted == y).sum().item()
    model.train()
    return correct / len(val_dataloader.dataset)


@ex.config
def config():
    n_epochs = 100  # noqa: F841
    lr = 1e-3  # noqa: F841
    lr_scheduler_kwargs = {  # noqa: F841
        "factor": 0.5,
        "patience": 10,
    }
    batch_size = 1024  # noqa: F841


@ex.main
def main(n_epochs, lr, lr_scheduler_kwargs, batch_size, _run):
    writer = SummaryWriter(Path(__file__).parents[2] / "saves" / "pretraining" / str(_run._id))

    in_memory = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_data, val_data = load_data(in_memory=in_memory, device=device)
    save_path = Path(__file__).parents[2] / "saves" / "pretraining"

    animations = boss_animations["DarkSoulsIII"]["iudex"]["all"]
    iudex_animations = [a for a in animations.values() if a["type"] != "movement"]
    n_classes = len(iudex_animations) + 1  # +1 for movement animations, see GameStateTransformer
    input_dims = train_data[0][0].shape
    output_dim = n_classes

    model = CNNEncoder(input_dims, output_dim).to(device)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, **lr_scheduler_kwargs)
    criterion = torch.nn.CrossEntropyLoss()

    n_workers = 0 if in_memory else 10
    train_dataloader = DataLoader(train_data, batch_size, num_workers=n_workers, shuffle=True)
    val_dataloader = DataLoader(val_data, batch_size, num_workers=n_workers, shuffle=True)

    losses, accuracies = [], []
    best_accuracy = 0

    pbar = tqdm(range(n_epochs))
    for i in range(n_epochs):
        epoch_loss = 0
        for x, y in train_dataloader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            y_pred = model(x)
            loss = criterion(y_pred, y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        accuracy = evaluate(model, val_dataloader, device)
        pbar.set_postfix({'Loss': epoch_loss, 'Accuracy': accuracy})
        accuracies.append(accuracy)
        pbar.update(1)
        losses.append(epoch_loss)
        writer.add_scalar("Train/Loss", epoch_loss, i)
        writer.add_scalar("Val/Accuracy", accuracy, i)
        writer.add_scalar("Train/Learning rate", optimizer.param_groups[0]["lr"], i)
        lr_scheduler.step(epoch_loss)
        if accuracy > best_accuracy:
            torch.save(model.state_dict(), save_path / "cnn_encoder_checkpoint.pt")
            best_accuracy = accuracy
    save_path.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), save_path / "cnn_encoder.pt")
    stats = {"losses": losses, "accuracies": accuracies}
    with open(save_path / "stats.json", "w") as f:
        json.dump(stats, f)


if __name__ == "__main__":
    sacred_dir = Path(__file__).parents[2] / "saves" / "pretraining"
    ex.observers.append(FileStorageObserver(sacred_dir))
    ex.run_commandline()
