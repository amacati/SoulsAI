"""The scheduler module contains schedulers to schedule changing hyperparameters during training.

The scheduler can be saved and loaded to allow for checkpointing of the complete training process.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from soulsai.utils import module_type_from_string

scheduler_cls = module_type_from_string(__name__)


class Scheduler(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self) -> torch.FloatTensor:
        """The value at the current step."""

    def update(self, x: int = 1):
        """Advance the scheduler.

        Args:
            x: Number of steps to advance the scheduler. Defaults to 1.
        """


class LinearScheduler(Scheduler):

    def __init__(self, start: list[float], end: list[float], steps: int):
        super().__init__()
        self.params = nn.ParameterDict({
            "start": nn.Parameter(torch.tensor(start), requires_grad=False),
            "end": nn.Parameter(torch.tensor(end), requires_grad=False),
            "steps": nn.Parameter(torch.tensor(steps), requires_grad=False),
            "step": nn.Parameter(torch.tensor(0), requires_grad=False),
        })

    def forward(self) -> torch.FloatTensor:
        """The value at the current decay step."""
        alpha = self.params["step"] / self.params["steps"]
        return self.params["start"] + (self.params["end"] - self.params["start"]) * alpha

    def update(self, n: int = 1):
        """Advance the scheduler.

        Args:
            n: Number of steps to advance the scheduler. Defaults to 1.
        """
        self.params["step"].copy_(torch.min(self.params["step"] + n, self.params["steps"]))
