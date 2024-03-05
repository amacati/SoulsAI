"""Visualization module for creating training plots from the training statistics."""
from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np

from soulsai.utils.utils import running_mean, running_std

if TYPE_CHECKING:
    from pathlib import Path


def save_plots(x: np.ndarray,
               ys: list[np.ndarray],
               xlabel: str,
               ylabels: list[str],
               path: Path,
               N_av: int = 50):
    """Plot and save the training progress dashboard.

    Stats are smoothed by computing the mean over a running window. Confidence intervals are given
    by computing the standard deviation over a running window.

    Args:
        x: Data for the x-axis.
        ys: Data for the y-axis. For each y, a running mean and standard deviation is computed.
        xlabel: Label for the x-axis.
        ylabels: Labels for the y-axis.
        path: Save location for the figure.
        N_av: Moving average window for smoothing the plot.
    """
    assert len(ys) == len(ylabels), "Number of y labels must match number of y data arrays"
    assert all(len(x) == len(y) for y in ys), "All y data arrays must have the same length"
    num_plots = len(ys)
    num_rows = num_plots // 2 + num_plots % 2
    fig, ax = plt.subplots(num_rows, 2, figsize=(15, 5 * num_rows))
    fig.suptitle("Training Statistics")
    for i, (y, ylabel) in enumerate(zip(ys, ylabels)):
        ax_i, ax_j = i // 2, i % 2
        y_mean = running_mean(y, N_av)
        y_std = running_std(y, N_av)
        ax[ax_i, ax_j].plot(x, y_mean)
        ax[ax_i, ax_j].fill_between(x, y_mean - y_std, y_mean + y_std, alpha=0.4)
        ax[ax_i, ax_j].set_title(ylabel)
        ax[ax_i, ax_j].set_xlabel(xlabel)
        ax[ax_i, ax_j].set_ylabel(ylabel)
        ax[ax_i, ax_j].grid(alpha=0.3)
        ax[ax_i, ax_j].set_xlabel(xlabel)
        ax[ax_i, ax_j].set_ylabel(ylabel)
        ax[ax_i, ax_j].set_title(ylabel)
        if len(x) > N_av:
            lim_low = min(y_mean[N_av:] - y_std[N_av:])
            # Add small value to avoid identical limits
            lim_up = max(y_mean[N_av:] + y_std[N_av:]) + 1e-4
            ax[ax_i, ax_j].set_ylim([lim_low - 0.1 * abs(lim_low), lim_up + 0.1 * abs(lim_up)])

    fig.savefig(path)
    plt.close(fig)
