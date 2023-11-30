"""The scheduler module contains an epsilon scheduler to decay epsilon values during training.

The scheduler can be saved and loaded to allow for checkpointing of the complete training process.
"""
from __future__ import annotations
import json
from typing import List, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from pathlib import Path


class EpsilonScheduler:
    """Scheduler class to linearly decay the epsilon value during training.

    Multiple decay sections with their own start, end and step values of epsilon are supported.
    """

    def __init__(self,
                 epsilon_max: List[float],
                 epsilon_min: List[float],
                 decay_steps: List[int],
                 zero_ending: bool = False):
        """Initialize the max, min and step arrays as numpy arrays.

        The max, min and step lists specify sections that are successively stepped through.
        Therefore, ``epsilon_max``, ``epsilon_min`` and ``decay_steps`` must be of the same length.

        Example:
            A decay from [0.5, 0.2] to [0.1, 0.01] with [10, 100] steps will first decay from 0.5 to
            0.1 over 10 steps, and then decays from 0.2 to 0.01 over 100 steps.

        Args:
            epsilon_max: Maximal epsilon values at the start of the decay.
            epsilon_min: Minimal epsilon values at the end of the decay.
            decay_steps: Number of decay steps.
            zero_ending: Flag to enable steps after the decay has finished (returning 0 values).
        """
        assert len(epsilon_max) == len(epsilon_min) == len(decay_steps)
        assert not any([x == 0 for x in epsilon_min])
        self._epsilon_max = np.array(epsilon_max)
        self._epsilon_min = np.array(epsilon_min)
        self._decay_steps = np.array(decay_steps)
        self._step = 0
        self._section = 0
        self._max_sections = len(epsilon_max)
        self._zero_ending = zero_ending

    @property
    def epsilon(self) -> float:
        """Epsilon value at the current decay step."""
        if self._zero_ending and self._section == self._max_sections:
            return 0
        assert self._section < self._max_sections
        return self._linear_decay(self._epsilon_max, self._epsilon_min, self._decay_steps,
                                  self._step)[self._section].copy()

    def step(self, n: int = 1):
        """Advance the scheduler.

        Args:
            n: Number of steps to advance the scheduler. Defaults to 1.
        """
        if self._zero_ending and self._section == self._max_sections:
            return
        if self._section == self._max_sections:
            raise ValueError("Scheduler has already finished!")
        if self._step + n > self._decay_steps[self._section]:
            n_steps = self._decay_steps[self._section] - self._step
            self._step = 0
            self._section += 1
            self.step(n - n_steps - 1)
        else:
            self._step += n

    @staticmethod
    def _linear_decay(epsilon_max: np.ndarray, epsilon_min: np.ndarray, decay_steps: np.ndarray,
                      current_step: float) -> np.ndarray:
        """Calculate the linearly decaying epsilon value at the current step.

        Args:
            epsilon_max: Maximal epsilon values at the start of the decay.
            epsilon_min: Minimal epsilon values at the end of the decay.
            decay_steps: Number of decay steps.
            current_step: Current decay step.

        Returns:
            The current epsilon value.
        """
        eps = epsilon_max - (epsilon_max - epsilon_min) * (current_step / decay_steps)
        return np.maximum(np.minimum(eps, epsilon_max), epsilon_min)

    def save(self, path: Path):
        """Save the scheduler to a json file.

        Args:
            path: Path to the json save file.
        """
        save = vars(self).copy()
        for key, val in save.items():  # Convert numpy arrays to lists for json
            if isinstance(val, np.ndarray):
                save[key] = val.tolist()
        with open(path, "w") as f:
            json.dump(save, f)

    def load(self, path: Path):
        """Load a scheduler from the specified path.

        Args:
            path: Path to the json save file.
        """
        with open(path, "r") as f:
            save = json.load(f)
        for key, val in save.items():  # Convert numpy arrays to lists for json
            if isinstance(val, List):
                val = np.array(val)
            setattr(self, key, val)
