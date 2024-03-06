"""The scheduler module contains schedulers to schedule changing hyperparameters during training.

Schedulers can be saved and loaded to allow for checkpointing of the complete training process.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from soulsai.utils import module_type_from_string

scheduler_cls = module_type_from_string(__name__)


class Scheduler(nn.Module):
    """Base class for schedulers.

    All schedulers should inherit from this class and implement the `forward` and `update` methods.
    """

    def __init__(self):
        """Initialize the scheduler."""
        super().__init__()

    def forward(self) -> torch.FloatTensor:
        """Calculate the value at the current step.

        Returns:
            The value at the current step.
        """

    def update(self, x: int = 1):
        """Advance the scheduler.

        Args:
            x: Number of steps to advance the scheduler. Defaults to 1.
        """


class LinearScheduler(Scheduler):
    """Linear scheduler to linearly change a value from start to end over a given number of steps.

    The scheduler can be used to decay hyperparameters during training, such as the exploration rate
    of an epsilon-greedy policy.
    """

    def __init__(self, start: list[float], end: list[float], steps: int):
        """Initialize the linear scheduler.

        Args:
            start: The start value of the scheduler.
            end: The end value of the scheduler.
            steps: The number of steps to reach the end value.
        """
        super().__init__()
        self.params = nn.ParameterDict({
            "start": nn.Parameter(torch.tensor(start), requires_grad=False),
            "end": nn.Parameter(torch.tensor(end), requires_grad=False),
            "steps": nn.Parameter(torch.tensor(steps), requires_grad=False),
            "step": nn.Parameter(torch.tensor(0), requires_grad=False),
        })

    def forward(self) -> torch.FloatTensor:
        """Calculate the value at the current decay step.

        Returns:
            The value at the current decay step.
        """
        alpha = self.params["step"] / self.params["steps"]
        return self.params["start"] + (self.params["end"] - self.params["start"]) * alpha

    def update(self, n: int = 1):
        """Advance the scheduler.

        Args:
            n: Number of steps to advance the scheduler. Defaults to 1.
        """
        self.params["step"].copy_(torch.min(self.params["step"] + n, self.params["steps"]))
