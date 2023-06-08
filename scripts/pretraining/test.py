from pathlib import Path
import matplotlib.pyplot as plt
import json


def main():
    p = Path(__file__).parents[2] / "saves" / "pretraining" / "stats.json"
    with open(p, "r") as f:
        stats = json.load(f)
    plt.plot(stats["losses"])
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.savefig(p.parent / "losses.png")
    plt.clf()
    plt.plot(stats["accuracies"])
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.savefig(p.parent / "accuracies.png")


if __name__ == "__main__":
    main()
