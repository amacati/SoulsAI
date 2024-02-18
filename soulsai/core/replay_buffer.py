"""The replay buffer module offers performant implementations of replay buffers for DQN and PPO."""
from __future__ import annotations

import sys
from abc import ABC, abstractmethod, abstractproperty
from typing import List, Dict, Type, TYPE_CHECKING, Callable
import random

import numpy as np
import torch
from tensordict import TensorDict

from soulsai.utils import module_type_from_string

if TYPE_CHECKING:
    from pathlib import Path

    from soulsai.core.agent import PPOAgent

buffer_cls: Callable[[str], type[AbstractBuffer]] = module_type_from_string(__name__)


class AbstractBuffer(ABC):
    """Abstract replay buffer class."""

    def __init__(self):
        """Initialize the buffer."""
        super().__init__()

    @abstractmethod
    def append(self, sample: Dict):
        """Append a sample to the buffer.

        Args:
            sample: Sample dictionary.
        """

    @abstractmethod
    def clear(self):
        """Clear the buffer from all samples."""

    @abstractmethod
    def __len__(self) -> int:
        """Get the length of the buffer.

        Returns:
            The buffer length.
        """

    @abstractproperty
    def filled(self) -> bool:
        """Check if the buffer is filled.

        Returns:
            True if the buffer is full, else false.
        """

    @abstractmethod
    def sample_batch(self, batch_size: int) -> list[np.ndarray]:
        """Sample a single batch from the buffer.

        Args:
            batch_size: Number of samples in the batch.

        Returns:
            The sampled batch.
        """

    @abstractmethod
    def sample_batches(self, batch_size: int, nbatches: int) -> list[list[np.ndarray]]:
        """Sample multiple batches from the buffer.

        If sufficient samples are available, the batches will not have dublicate samples across all
        batches.

        Args:
            batch_size: Number of samples per batch.
            nbatches: Number of batches.

        Returns:
            The sampled batches.

        Raises:
            RuntimeError: Asked to sample more samples per batch than currently available.
        """

    def save(self, path: Path):
        """Save the buffers to the specified path.

        Uses the torch save function to save a dictionary of the tensors.

        Args:
            path: The save file path.
        """
        torch.save(self.__dict__, path)

    def load(self, path: Path):
        """Load the buffers from the file.

        Args:
            path: The save file path.
        """
        self.__dict__.clear()
        self.__dict__.update(torch.load(path))


class ReplayBuffer(AbstractBuffer):
    """Implementation of a replay buffer that lazily allocates storage.

    Buffers for samples are allocated on receiving the first sample. An internal index keeps track
    of the current size of the buffer and enables to only sample from the parts of the buffers
    already filled with experience.
    """

    def __init__(self,
                 max_size: int,
                 device: torch.device = torch.device("cpu"),
                 seed: int | None = None):
        """Create the buffer tensor dict and set the index to 0.

        Args:
            max_size: Maximum buffer capacity.
            device: Buffer device.
            seed: Random seed to control the sampling.
        """
        super().__init__()
        self.max_size = max_size
        self.device = device
        self.buffer: TensorDict[torch.Tensor] = TensorDict({}, batch_size=max_size, device=device)
        # Helper indices to implement a ring buffer
        self._idx = 0
        self._maxidx = -1
        # Reproducible random number generator
        self.rng = np.random.default_rng(seed=seed)
        random.seed(seed)

    def append(self, sample: TensorDict[torch.Tensor]):
        """Append a sample to the buffer.

        Args:
            sample: Sample dictionary containing the observation, action, reward etc.
        """
        num_samples = sample.batch_size[0]
        assert num_samples < self.max_size, "Sample size must be smaller than the buffer"
        self._allocate_buffers(sample)
        # If num_samples + self._idx > self.max_size, the index wraps around to the beginning of the
        # buffer. We take the index vector modulo ``self.max_size`` to implement this behavior.
        idx = torch.arange(self._idx, self._idx + num_samples) % self.max_size
        self.buffer[idx] = sample
        # Update the helper indices
        self._idx = (self._idx + num_samples) % self.max_size
        self._maxidx = min(self._maxidx + num_samples, self.max_size - 1)

    def _allocate_buffers(self, sample: TensorDict[torch.Tensor]):
        # Check if there are unknown keys in the sample and allocate buffers for them if necessary
        for key in set(sample.keys()).difference(set(self.buffer.keys())):
            if isinstance(sample[key], torch.Tensor):
                self.buffer[key] = torch.empty((self.max_size, *sample[key].shape[1:]),
                                               dtype=sample[key].dtype,
                                               device=self.device)
            elif isinstance(sample[key], TensorDict):
                self.buffer[key] = TensorDict(sample[key],
                                              batch_size=self.max_size,
                                              device=self.device)
            else:
                raise ValueError(f"Unknown sample type for key {key}")

    def clear(self):
        """Clear the buffer from all samples."""
        self._idx = 0
        self._maxidx = -1

    def __len__(self) -> int:
        """Get the length of the buffer.

        Returns:
            The buffer length.
        """
        return self._maxidx + 1

    @property
    def filled(self) -> bool:
        """Check if the buffer is filled.

        Returns:
            True if the buffer is full, else false.
        """
        return self._maxidx + 1 == self.max_size

    def sample_batch(self, batch_size: int) -> TensorDict[torch.Tensor]:
        """Sample a single batch from the buffer.

        Args:
            batch_size: Number of samples in the batch.

        Returns:
            The sampled batch.

        Raises:
            RuntimeError: Asked to sample more samples than currently available.
        """
        if batch_size > self._maxidx + 1:
            raise RuntimeError("Asked to sample more elements than available in buffer")
        return self.buffer[np.array(random.sample(range(self._maxidx + 1), batch_size))]

    def sample_batches(self, batch_size: int, nbatches: int) -> List[TensorDict[torch.Tensor]]:
        """Sample multiple batches from the buffer.

        If sufficient samples are available, the batches will not have dublicate samples across all
        batches.

        Args:
            batch_size: Number of samples per batch.
            nbatches: Number of batches.

        Returns:
            The sampled batches.

        Raises:
            RuntimeError: Asked to sample more samples per batch than currently available.
        """
        if batch_size > self._maxidx + 1:
            raise RuntimeError("Asked to sample more elements than available in buffer")
        # If the buffer contains more samples than requested in total, indices are chosen such that
        # no sample is sampled twice across all batches. If more total samples are requested than
        # available in the buffer, resort to random independent indices in each batch
        nsamples = batch_size * nbatches
        if nsamples <= self._maxidx + 1:
            i = np.array(random.sample(range(self._maxidx + 1), nsamples))
        else:
            i = self.rng.integers(0, self._maxidx + 1, size=nsamples)
        return self.buffer[i].reshape(nbatches, batch_size, -1)


class PrioritizedReplayBuffer(AbstractBuffer):
    """Implementation of a prioritized replay buffer.

    Buffers for observations, actions, action masks, rewards, next observations, and terminations
    are preallocated. An internal index keeps track of the current size of the buffer and enables to
    only sample from the parts of the buffers already filled with experience.

    We fix alpha to 0.5 and use sqrt instead of **alpha. See e.g. Dopamine implementation at
    https://github.com/google/dopamine/blob/a6f414ca01a81e933359a4922965178a40e0f38a/dopamine/jax/agents/quantile/quantile_agent.py#L262
    """

    def __init__(self,
                 maxlen: int,
                 obs_shape: tuple[int, ...],
                 n_actions: int,
                 action_masking: bool = False,
                 beta: float = 0.5,
                 obs_dtype: str | np.dtype = np.float32):
        """Preallocate the buffer arrays and set the index to 0.

        Args:
            maxlen: Maximum buffer capacity.
            obs_shape: Observation shape.
            n_actions: Number of possible actions.
            action_masking: Flag to disable/enable action masking.
            beta: Weight correction exponent. Controls how much bias correction is used.
        """
        super().__init__()
        self.maxlen = maxlen
        self._idx = 0
        self._maxidx = -1
        self.buffers = {
            "obs": np.zeros((maxlen, *obs_shape), dtype=obs_dtype),
            "action": np.zeros(maxlen, dtype=np.int64),
            "action_mask": np.zeros((maxlen, n_actions)),
            "reward": np.zeros(maxlen),
            "nextObs": np.zeros((maxlen, *obs_shape), dtype=obs_dtype),
            "terminated": np.zeros(maxlen),
            "truncated": np.zeros(maxlen),
            "priority": np.zeros(maxlen)
        }
        self._action_masking = action_masking
        self._sum_priorities_alpha = 0
        self._max_priority_idx = 0
        self.beta = beta

    def append(self, sample: Dict):
        """Append a sample to the buffer.

        Args:
            sample: DQN sample dict containing the observation, the action, the reward, the next
            observation, and the terminated flag. If action masking is used, the info dict must
            contain a key "allowed_actions" with the list of allowed actions.
        """
        for key in ("obs", "action", "reward", "nextObs", "terminated", "truncated"):
            self.buffers[key][self._idx] = sample[key]
        if self._action_masking:
            self.buffers["action_mask"][self._idx] = 0
            self.buffers["action_mask"][self._idx, sample["info"]["allowed_actions"]] = 1
        # Update the sum of priorities (to the power of alpha) and the maximum priority index. We
        # keep track of the sum of priorities to avoid having to recompute it every time we sample
        # from the buffer.
        self._sum_priorities_alpha -= np.sqrt(self.buffers["priority"][self._idx])
        # Check if the maximum index is about to be overwritten. If it is, we first need to
        # recompute the new maximum index and priority. In practice, this does not happen often, as
        # older samples should generally have a lower TD error and therefore lower priority.
        if self._idx == self._max_priority_idx:
            self.buffers["priority"][self._idx] = 0.  # Set to 0 to remove from max calculation
            self._max_priority_idx = np.argmax(self.buffers["priority"])
        # Set the priortiy of the new sample to the current maximum priority and update the sum of
        # priorities
        priority = 1.0 if self._maxidx == -1 else self.buffers["priority"][self._max_priority_idx]
        self.buffers["priority"][self._idx] = priority + 1e-10
        self._sum_priorities_alpha += np.sqrt(self.buffers["priority"][self._idx])
        # Update the internal index and the maximum index
        self._idx = (self._idx + 1) % self.maxlen
        self._maxidx = min(self._maxidx + 1, self.maxlen - 1)

    def clear(self):
        """Clear the buffer from all samples."""
        self._idx = 0
        self._maxidx = -1

    def __len__(self) -> int:
        """Get the length of the buffer.

        Returns:
            The buffer length.
        """
        return self._maxidx + 1

    @property
    def filled(self) -> bool:
        """Check if the buffer is filled.

        Returns:
            True if the buffer is full, else false.
        """
        return self._maxidx + 1 == self.maxlen

    def sample_batch(self, batch_size: int) -> list[np.ndarray]:
        """Sample a single batch from the buffer.

        Args:
            batch_size: Number of samples in the batch.

        Returns:
            The sampled batch including the weights and the indices of the samples.

        Raises:
            RuntimeError: Asked to sample more samples than currently available.
        """
        if batch_size > self._maxidx + 1:
            raise RuntimeError("Asked to sample more elements than available in buffer")
        priorities = self.buffers["priority"][:self._maxidx + 1]
        probabilities = np.sqrt(priorities) / self._sum_priorities_alpha
        i = np.random.choice(self._maxidx + 1, batch_size, replace=False, p=probabilities)
        keys = ("obs", "action", "reward", "nextObs", "terminated")  # Order of keys is important!
        if self._action_masking:
            keys += ("action_mask",)
        batch = [self.buffers[key][i] for key in keys]
        weights = ((self._maxidx + 1) * probabilities[i])**-self.beta
        normalized_weights = weights / weights.max()
        batch.extend([normalized_weights.astype(np.float32), i])
        return batch

    def sample_batches(self, batch_size: int, nbatches: int) -> list[list[np.ndarray]]:
        """Sample multiple batches from the buffer.

        If sufficient samples are available, the batches will not have dublicate samples across all
        batches.

        Args:
            batch_size: Number of samples per batch.
            nbatches: Number of batches.

        Returns:
            The sampled batches including the weights and the indices of the samples.

        Raises:
            RuntimeError: Asked to sample more samples per batch than currently available.
        """
        if batch_size > self._maxidx + 1:
            raise RuntimeError("Asked to sample more elements than available in buffer")
        priorities = self.buffers["priority"][:self._maxidx + 1]
        probabilities = np.sqrt(priorities) / self._sum_priorities_alpha
        # If the buffer contains more samples than requested in total, indices are chosen such that
        # no sample is sampled twice across all batches. If more total samples are requested than
        # available in the buffer, resort to random independent indices in each batch
        if batch_size * nbatches <= self._maxidx + 1:
            nsamples = batch_size * nbatches
            unique_indices = np.random.choice(self._maxidx + 1,
                                              nsamples,
                                              replace=False,
                                              p=probabilities)
            indices = np.split(unique_indices, nbatches)
        else:
            indices = [
                np.random.choice(self._maxidx + 1, batch_size, replace=False, p=probabilities)
                for _ in range(nbatches)
            ]
        weights = [((self._maxidx + 1) * probabilities[i])**-self.beta for i in indices]
        normalized_weights = [(w / w.max()).astype(np.float32) for w in weights]
        keys = ("obs", "action", "reward", "nextObs", "terminated")  # Order of keys is important!
        if self._action_masking:
            keys += ("action_mask",)
        batches = [[self.buffers[key][i] for key in keys] for i in indices]
        # Add the corresponding weights and indices to each batch
        batches = [
            batch + [weight, i] for batch, weight, i in zip(batches, normalized_weights, indices)
        ]
        return batches

    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray):
        """Update the priorities of the samples at the specified indices.

        Args:
            indices: The indices of the samples to update.
            priorities: The new priorities of the samples.
        """
        # First, check if the maximum has changed. If so, update the maximum index. Then, update the
        # sum of priorities by subtracting the old priorities and adding the new ones to the sum.
        # Finally, update the priorities in the buffer.
        if np.max(priorities) > self.buffers["priority"][self._max_priority_idx]:
            self._max_priority_idx = indices[np.argmax(priorities)]
        self._sum_priorities_alpha -= np.sum(np.sqrt(self.buffers["priority"][indices]))
        self.buffers["priority"][indices] = priorities + 1e-10
        self._sum_priorities_alpha += np.sum(np.sqrt(self.buffers["priority"][indices]))


class TrajectoryBuffer:
    """Experience buffer to hold samples of multiple trajectories while preserving their order.

    The buffer has a fixed amount of trajectories with a fixed amount of samples. Each sample has a
    unique trajectory ID and a step ID. The buffer is full when the samples from all trajectories
    for all steps have been appended. Samples can be appended out of order since the IDs are used to
    sort them into the correct order.

    Note:
        This buffer is designed for categorical actions PPO only!
    """

    def __init__(self, n_trajectories: int, n_samples: int, obs_shape: tuple[int, ...]):
        """Preallocate the sample tensors and completion flags.

        Args:
            n_trajectories: Number of parallel trajectories from parallel environments used.
            n_samples: Number of samples for each trajectory.
            obs_shape: Observation shape.
        """
        self.n_trajectories = n_trajectories
        self.n_samples = n_samples
        self.n_batch_samples = n_trajectories * n_samples
        self.obs = torch.zeros((n_trajectories * n_samples, *obs_shape), dtype=torch.float32)
        self.actions = torch.zeros((n_trajectories * n_samples, 1), dtype=torch.int64)
        self.probs = torch.zeros((n_trajectories * n_samples, 1), dtype=torch.float32)
        self.rewards = torch.zeros(n_trajectories * n_samples, dtype=torch.float32)
        self.terminated = torch.zeros((n_trajectories * n_samples, 1), dtype=torch.float32)
        self.values = torch.zeros((n_trajectories * n_samples, 1), dtype=torch.float32)
        self.advantages = torch.zeros((n_trajectories * n_samples, 1), dtype=torch.float32)
        # For the advantage calculation, we need the next observation after the final sample.
        self._final_obs = torch.zeros((n_trajectories, *obs_shape), dtype=torch.float32)
        self._end_terminated = torch.zeros((n_trajectories, 1), dtype=torch.float32)
        self._end_values = torch.zeros((n_trajectories, 1), dtype=torch.float32)
        # Create an array that tracks which samples have already been added for fast checking
        self._complete_flags = np.empty(n_trajectories * (n_samples + 1), dtype=np.bool_)
        self._complete_flags[:] = False

    def append(self, sample: Dict):
        """Append a PPO sample to the buffer.

        Also sets the complete flag for the received sample.

        Args:
            sample: PPO sample consisting of the observation, the chosen action, the action
            probability, the reward, the terminated flag, the trajectory ID and the step ID.
        """
        trajectory_id, step_id = sample["clientId"], sample["stepId"]
        assert trajectory_id < self.n_trajectories and step_id <= self.n_samples
        if step_id != self.n_samples:  # Non-terminal sample
            idx = trajectory_id * self.n_samples + step_id
            self.obs[idx, :] = torch.from_numpy(sample["obs"])
            self.actions[idx] = sample["action"]
            self.probs[idx] = sample["prob"]
            self.rewards[idx] = sample["reward"]
            self.terminated[idx] = sample["terminated"]
        else:
            self._final_obs[trajectory_id] = torch.from_numpy(sample["obs"])
            self._end_terminated[trajectory_id] = sample["terminated"]
        self._complete_flags[trajectory_id * (self.n_samples + 1) + step_id] = True

    def clear(self):
        """Clear the buffer complete flags and reset the advantage values."""
        self._complete_flags[:] = False
        self.advantages[:] = torch.nan  # Make sure to not accidentally use old advantages

    @property
    def buffer_complete(self) -> bool:
        """Flag to check if all required samples have been added to the buffer."""
        return np.all(self._complete_flags)

    def compute_advantages_and_values(self, agent: PPOAgent, gamma: float, gae_lambda: float):
        """Compute the advantage of each observation with GAE and save the values in the buffer.

        Warning:
            This function HAS to be called before using the samples for training.

        Args:
            agent: The current PPO agent.
            gamma: The discount factor.
            gae_lambda: The GAE discount factor. A lower value is only gamma-just for an accurate
                estimate of the value function, but reduces the estimate's variance.
        """
        self.values = agent.get_values(self.obs, requires_grad=False).cpu()
        self._end_values = agent.get_values(self._final_obs, requires_grad=False).cpu()
        for trajectory_id in range(self.n_trajectories):
            last_advantage = 0
            for step_id in reversed(range(self.n_samples)):
                # The estimation computes the advantage values in reverse order by using the
                # value and advantage estimate of time t + 1 for the observation at time t
                idx = trajectory_id * self.n_samples + step_id
                if step_id == self.n_samples - 1:  # Terminal sample. Use actual end values
                    not_terminated = (1. - self._end_terminated[trajectory_id])
                    next_value = self._end_values[trajectory_id]
                else:
                    not_terminated = (1. - self.terminated[idx])
                    next_value = self.values[idx + 1]
                td_target = self.rewards[idx] + gamma * next_value * not_terminated
                td_error = td_target - self.values[idx]
                last_advantage = td_error + gamma * gae_lambda * not_terminated * last_advantage
                self.advantages[idx] = last_advantage
