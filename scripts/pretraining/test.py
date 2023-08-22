import time
from pathlib import Path

import cv2
import einops
import torch
import numpy as np
from torchvision.transforms.functional import normalize, to_tensor

from soulsgym.games import DarkSoulsIII

from soulsai.core.models import AutoEncoder


def test():
    model = AutoEncoder(2048)
    x = torch.randn(2, 3, 90, 160)
    print(model(x).shape)


def test2():
    print(int(np.sqrt(2048 / (16 * 9)) * 16), int(np.sqrt(2048 / (16 * 9)) * 9))


@torch.no_grad()
def main():
    game = DarkSoulsIII()
    # model = AutoEncoder(2048)
    # encoder_dict = torch.load(
    #     Path(__file__).parents[2] / "saves/autoencoder/5/checkpoints/best_autoencoder.pt")
    # model.load_state_dict(encoder_dict)
    # model.cuda().eval()
    cv2.namedWindow("DarkSoulsIIIAE", cv2.WINDOW_NORMAL)
    # data_path = Path(__file__).parents[2] / "data" / "soulsgym_dataset" / "train_data"
    # mean, std = torch.load(data_path / "mean_std.pt")
    while True:
        t_start = time.perf_counter()
        # x = normalize(to_tensor(game.img), mean, std)
        # img = model(x.cuda().unsqueeze(0))[0].cpu().numpy()
        # img = einops.rearrange(img, "c h w -> h w c")
        img = cv2.cvtColor(game.img[..., [2, 1, 0]], cv2.COLOR_BGR2GRAY)
        img = (cv2.resize(img, (40, 23), interpolation=cv2.INTER_AREA) * 1.5) / 255. - 0.2
        cv2.imshow("DarkSoulsIIIAE", img)
        cv2.waitKey(1)
        t_end = time.perf_counter()
        time.sleep(max(0, 1 / 60 - (t_end - t_start)))


if __name__ == "__main__":
    # test()
    # test2()
    main()
