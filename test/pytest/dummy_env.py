from pathlib import Path

import pandas as pd
import gymnasium


class IudexDummyEnv(gymnasium.Env):

    def __init__(self, data_path: Path):
        super().__init__()
        self.data = pd.read_csv(data_path)

    def reset(self):
        ...

    def step(self, action):
        ...
