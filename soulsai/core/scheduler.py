from pathlib import Path
import json
from typing import List

import numpy as np


class EpsilonScheduler:

    def __init__(self, epsilon_max, epsilon_min, decay_steps, zero_ending=False):
        assert len(epsilon_max) == len(epsilon_min) == len(decay_steps)
        self._epsilon_max = np.array(epsilon_max)
        self._epsilon_min = np.array(epsilon_min)
        self._decay_steps = np.array(decay_steps)
        self._step = 0
        self._section = 0
        self._max_sections = len(epsilon_max)
        self._zero_ending = zero_ending

    @property
    def epsilon(self):
        if self._zero_ending and self._section == self._max_sections:
            return 0
        assert self._section < self._max_sections
        return self._exponential_decay(self._epsilon_max, self._epsilon_min, self._decay_steps,
                                       self._step)[self._section]

    def step(self):
        if self._zero_ending and self._section == self._max_sections:
            return
        if self._step == self._decay_steps[self._section]:
            self._step = 0
            self._section += 1
        else:
            self._step += 1

    @staticmethod
    def _exponential_decay(epsilon_max, epsilon_min, decay_steps, current_step):
        eps = epsilon_max*(epsilon_min/(epsilon_max + 1e-8))**(current_step/decay_steps)
        return np.maximum(epsilon_min, eps)

    def save(self, path: Path):
        save = vars(self).copy()
        for key, val in save.items():  # Convert numpy arrays to lists for json
            if isinstance(val, np.ndarray):
                save[key] = val.tolist()
        with open(path, "w") as f:
            json.dump(save, f)

    def load(self, path: Path):
        with open(path, "r") as f:
            save = json.load(f)
        for key, val in save.items():  # Convert numpy arrays to lists for json
            if isinstance(val, List):
                val = np.array(val)
            setattr(self, key, val)
