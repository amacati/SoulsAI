import time
from soulsgym.core.game_interface import Game
from soulsgym.core.logger import Logger
from soulsgym.envs.iudex_env import IudexEnv
from visualization import save_plots
import numpy as np
from pathlib import Path


if __name__ == "__main__":
    rewards = np.arange(100) + np.random.rand(100)
    steps = np.random.rand(100)*10 + 200
    hp = np.random.rand(100) * 1037
    win = [True if np.random.rand() > 0.5 else False for _ in range(100)]
    save_plots(rewards, steps, hp, win, Path(__file__).parent / "training_test.png")
