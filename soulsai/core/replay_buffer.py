"""The replay buffer module offers performant implementations of replay buffers for DQN and PPO."""
from __future__ import annotations

from pathlib import Path
from typing import List, Dict, Type
import sys

import numpy as np
import torch

from soulsai.core.agent import PPOAgent


def get_buffer_class(buffer_type: str) -> Type[ReplayBuffer | PrioritizedReplayBuffer]:
    """Get the buffer class from the buffer string.

    Note:
        This function returns a type rather than an instance!

    Args:
        buffer_type: The buffer type name.

    Returns:
        The buffer type.

    Raises:
        AttributeError: The specified buffer type does not exist.
    """
    return getattr(sys.modules[__name__], buffer_type)


class ReplayBuffer:
    """Fast implementation of a replay buffer.

    Buffers for states, actions, action masks, rewards, next states, and dones are preallocated.
    An internal index keeps track of the current size of the buffer and enables to only sample from
    the parts of the buffers already filled with experience.
    """

    def __init__(self, maxlen: int, n_states: int, n_actions: int, action_masking: bool = False):
        """Preallocate the buffer arrays and set the index to 0.

        Args:
            maxlen: Maximum buffer capacity.
            n_states: State dimensionality.
            n_actions: Number of possible actions.
            action_masking: Flag to disable/enable action masking.
        """
        self.maxlen = maxlen
        self._idx = 0
        self._maxidx = -1
        self._b_s = np.zeros((maxlen, n_states))
        self._b_a = np.zeros(maxlen, dtype=np.int64)
        self._b_am = np.zeros((maxlen, n_actions))
        self._b_r = np.zeros(maxlen)
        self._b_sn = np.zeros((maxlen, n_states))
        self._b_d = np.zeros(maxlen)
        self._action_masking = action_masking

    def append(self, sample: Dict):
        """Append a sample to the buffer.

        Args:
            sample: DQN sample dict containing the observation, the action, the reward, the next
            observation, and the done flag. If action masking is used, the info dict must contain a
            key "allowed_actions" with the list of allowed actions.
        """
        self._b_s[self._idx] = sample["obs"]
        self._b_a[self._idx] = sample["action"]
        self._b_r[self._idx] = sample["reward"]
        self._b_sn[self._idx] = sample["nextObs"]
        self._b_d[self._idx] = sample["done"]
        if self._action_masking:
            self._b_am[self._idx] = 0
            self._b_am[self._idx, sample["info"]["allowed_actions"]] = 1
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

    def sample_batch(self, batch_size: int) -> List[np.ndarray]:
        """Sample a single batch from the buffer.

        Args:
            batch_size: Number of samples in the batch.
        """
        if batch_size > self._maxidx + 1:
            raise RuntimeError("Asked to sample more elements than available in buffer")
        i = np.random.choice(self._maxidx + 1, batch_size, replace=False)
        if self._action_masking:
            return [
                self._b_s[i], self._b_a[i], self._b_r[i], self._b_sn[i], self._b_d[i], self._b_am[i]
            ]  # noqa: E501
        return [self._b_s[i], self._b_a[i], self._b_r[i], self._b_sn[i], self._b_d[i]]

    def sample_batches(self, batch_size: int, nbatches: int) -> List[np.ndarray]:
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
        if batch_size * nbatches <= self._maxidx + 1:
            nsamples = batch_size * nbatches
            unique_indices = np.random.choice(self._maxidx + 1, nsamples, replace=False)
            indices = np.split(unique_indices, nbatches)
        else:
            indices = [
                np.random.choice(self._maxidx + 1, batch_size, replace=False)
                for _ in range(nbatches)
            ]
        if self._action_masking:
            batches = [[
                self._b_s[i], self._b_a[i], self._b_r[i], self._b_sn[i], self._b_d[i], self._b_am[i]
            ] for i in indices]
        else:
            batches = [[self._b_s[i], self._b_a[i], self._b_r[i], self._b_sn[i], self._b_d[i]]
                       for i in indices]
        return batches

    def save(self, path: Path):
        """Save the buffers to the specified path.

        Uses the torch save function to save a dictionary of the tensors.

        Args:
            path: The save file path.
        """
        save_dict = {
            "_b_s": self._b_s,
            "_b_a": self._b_a,
            "_b_r": self._b_r,
            "_b_sn": self._b_sn,
            "_b_d": self._b_d,
            "_idx": self._idx,
            "_maxidx": self._maxidx
        }
        if self._action_masking:
            save_dict["_b_am"] = self._b_am
        torch.save(save_dict, path)

    def load(self, path: Path):
        """Load the buffers from the file.

        Args:
            path: The save file path.
        """
        save_dict = torch.load(path)
        for att in ("_b_s", "_b_a", "_b_r", "_b_sn", "_b_d", "_idx", "_maxidx"):
            setattr(self, att, save_dict[att])
        if self._action_masking:
            self._b_am = save_dict["_b_am"]
        self.maxlen = self._b_s.shape[0]


class TrajectoryBuffer:
    """Experience buffer to hold samples of multiple trajectories while preserving their order.

    The buffer has a fixed amount of trajectories with a fixed amount of samples. Each sample has a
    unique trajectory ID and a step ID. The buffer is full when the samples from all trajectories
    for all steps have been appended. Samples can be appended out of order since the IDs are used to
    sort them into the correct order.

    Note:
        This buffer is designed for categorical actions PPO only!
    """

    def __init__(self, n_trajectories: int, n_samples: int, n_states: int):
        """Preallocate the sample tensors and completion flags.

        Args:
            n_trajectories: Number of parallel trajectories from parallel environments used.
            n_samples: Number of samples for each trajectory.
            n_states: State dimensionality.
        """
        self.n_trajectories = n_trajectories
        self.n_samples = n_samples
        self.n_batch_samples = n_trajectories * n_samples
        self.n_states = n_states
        self.states = torch.zeros((n_trajectories * n_samples, n_states), dtype=torch.float32)
        self.actions = torch.zeros((n_trajectories * n_samples, 1), dtype=torch.int64)
        self.probs = torch.zeros((n_trajectories * n_samples, 1), dtype=torch.float32)
        self.rewards = torch.zeros(n_trajectories * n_samples, dtype=torch.float32)
        self.dones = torch.zeros((n_trajectories * n_samples, 1), dtype=torch.float32)
        self.values = torch.zeros((n_trajectories * n_samples, 1), dtype=torch.float32)
        self.advantages = torch.zeros((n_trajectories * n_samples, 1), dtype=torch.float32)
        # For the advantage calculation, we need the next state after the final sample.
        self._end_states = torch.zeros((n_trajectories, n_states), dtype=torch.float32)
        self._end_dones = torch.zeros((n_trajectories, 1), dtype=torch.float32)
        self._end_values = torch.zeros((n_trajectories, 1), dtype=torch.float32)
        # Create an array that tracks which samples have already been added for fast checking
        self._complete_flags = np.empty(n_trajectories * (n_samples + 1), dtype=np.bool_)
        self._complete_flags[:] = False

    def append(self, sample: Dict):
        """Append a PPO sample to the buffer.

        Also sets the complete flag for the received sample.

        Args:
            sample: PPO sample consisting of the observation, the chosen action, the action
            probability, the reward, the done flag, the trajectory ID and the step ID.
        """
        trajectory_id, step_id = sample["clientId"], sample["stepId"]
        assert trajectory_id < self.n_trajectories and step_id <= self.n_samples
        if step_id != self.n_samples:  # Non-terminal sample
            idx = trajectory_id * self.n_samples + step_id
            self.states[idx, :] = torch.from_numpy(sample["obs"])
            self.actions[idx] = sample["action"]
            self.probs[idx] = sample["prob"]
            self.rewards[idx] = sample["reward"]
            self.dones[idx] = sample["done"]
        else:
            self._end_states[trajectory_id] = torch.from_numpy(sample["obs"])
            self._end_dones[trajectory_id] = sample["done"]
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
        """Compute the advantage of each state with GAE and save the values in the value buffer.

        Warning:
            This function HAS to be called before using the samples for training.

        Args:
            agent: The current PPO agent.
            gamma: The discount factor.
            gae_lambda: The GAE discount factor. A lower value is only gamma-just for an accurate
                estimate of the value function, but reduces the estimate's variance.
        """
        self.values = agent.get_values(self.states, requires_grad=False)
        self._end_values = agent.get_values(self._end_states, requires_grad=False)
        for trajectory_id in range(self.n_trajectories):
            last_advantage = 0
            for step_id in reversed(range(self.n_samples)):
                # The estimation computes the advantage values in reverse order by using the
                # value and advantage estimate of time t + 1 for the state at time t
                idx = trajectory_id * self.n_samples + step_id
                if step_id == self.n_samples - 1:  # Terminal sample. Use actual end values
                    not_done = (1. - self._end_dones[trajectory_id])
                    next_value = self._end_values[trajectory_id]
                else:
                    not_done = (1. - self.dones[idx])
                    next_value = self.values[idx + 1]
                td_error = self.rewards[idx] + gamma * next_value * not_done - self.values[idx]
                last_advantage = td_error + gamma * gae_lambda * not_done * last_advantage
                self.advantages[idx] = last_advantage


class PrioritizedReplayBuffer:
    """Implementation of a prioritized replay buffer.

    Buffers for states, actions, action masks, rewards, next states, and dones are preallocated.
    An internal index keeps track of the current size of the buffer and enables to only sample from
    the parts of the buffers already filled with experience.
    """

    def __init__(self,
                 maxlen: int,
                 n_states: int,
                 n_actions: int,
                 action_masking: bool = False,
                 beta: float = 0.5):
        """Preallocate the buffer arrays and set the index to 0.

        Args:
            maxlen: Maximum buffer capacity.
            n_states: State dimensionality.
            n_actions: Number of possible actions.
            action_masking: Flag to disable/enable action masking.
            beta: Weight correction exponent. Controls how much bias correction is used.
        """
        self.maxlen = maxlen
        self._idx = 0
        self._maxidx = -1
        self._b_s = np.zeros((maxlen, n_states))
        self._b_a = np.zeros(maxlen, dtype=np.int64)
        self._b_am = np.zeros((maxlen, n_actions))
        self._b_r = np.zeros(maxlen)
        self._b_sn = np.zeros((maxlen, n_states))
        self._b_d = np.zeros(maxlen)
        self._action_masking = action_masking
        self._priorities = np.zeros(maxlen)
        self._sum_priorities_alpha = 0
        self._max_priority_idx = 0
        self.beta = beta

    def append(self, sample: Dict):
        """Append a sample to the buffer.

        Args:
            sample: DQN sample dict containing the observation, the action, the reward, the next
            observation, and the done flag. If action masking is used, the info dict must contain a
            key "allowed_actions" with the list of allowed actions.
        """
        self._b_s[self._idx] = sample["obs"]
        self._b_a[self._idx] = sample["action"]
        self._b_r[self._idx] = sample["reward"]
        self._b_sn[self._idx] = sample["nextObs"]
        self._b_d[self._idx] = sample["done"]
        if self._action_masking:
            self._b_am[self._idx] = 0
            self._b_am[self._idx, sample["info"]["allowed_actions"]] = 1
        # Update the sum of priorities (to the power of alpha) and the maximum priority index. We
        # keep track of the sum of priorities to avoid having to recompute it every time we sample
        # from the buffer.
        self._sum_priorities_alpha -= self._priorities[self._idx]
        # Check if the maximum index is about to be overwritten. If it is, we first need to
        # recompute the new maximum index and priority. In practice, this does not happen often, as
        # older samples should generally have a lower TD error and therefore lower priority.
        if self._idx == self._max_priority_idx:
            self._priorities[self._idx] = 0.  # Set to 0 to remove from max calculation
            self._max_priority_idx = np.argmax(self._priorities)
        # Set the priortiy of the new sample to the current maximum priority and update the sum of
        # priorities
        priority = 1.0 if self._maxidx == -1 else self._priorities[self._max_priority_idx]
        self._priorities[self._idx] = priority + 1e-10
        self._sum_priorities_alpha += np.sqrt(self._priorities[self._idx])
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

    def sample_batch(self, batch_size: int) -> tuple[list[np.ndarray], np.ndarray, np.ndarray]:
        """Sample a single batch from the buffer.

        Args:
            batch_size: Number of samples in the batch.

        Returns:
            The sampled batch, the indices of the samples, and the weights of the samples.
        """
        if batch_size > self._maxidx + 1:
            raise RuntimeError("Asked to sample more elements than available in buffer")
        priorities = self._priorities[:self._maxidx + 1]
        probabilities = np.sqrt(priorities) / self._sum_priorities_alpha
        i = np.random.choice(self._maxidx + 1, batch_size, replace=False, p=probabilities)
        sample = [self._b_s[i], self._b_a[i], self._b_r[i], self._b_sn[i], self._b_d[i]]
        if self._action_masking:
            sample.append(self._b_am[i])
        weights = ((self._maxidx + 1) * probabilities[i])**-self.beta
        normalized_weights = weights / weights.max()
        return sample, i, normalized_weights.astype(np.float32)

    def sample_batches(
            self, batch_size: int,
            nbatches: int) -> tuple[list[list[np.ndarray]], list[np.ndarray], list[np.ndarray]]:
        """Sample multiple batches from the buffer.

        If sufficient samples are available, the batches will not have dublicate samples across all
        batches.

        Args:
            batch_size: Number of samples per batch.
            nbatches: Number of batches.

        Returns:
            The sampled batches, the sample indices and their weights.

        Raises:
            RuntimeError: Asked to sample more samples per batch than currently available.
        """
        if batch_size > self._maxidx + 1:
            raise RuntimeError("Asked to sample more elements than available in buffer")
        priorities = self._priorities[:self._maxidx + 1]
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
        if self._action_masking:
            batches = [[
                self._b_s[i], self._b_a[i], self._b_r[i], self._b_sn[i], self._b_d[i], self._b_am[i]
            ] for i in indices]
        else:
            batches = [[self._b_s[i], self._b_a[i], self._b_r[i], self._b_sn[i], self._b_d[i]]
                       for i in indices]
        weights = [((self._maxidx + 1) * probabilities[i])**-self.beta for i in indices]
        normalized_weights = [(w / w.max()).astype(np.float32) for w in weights]
        return batches, indices, normalized_weights

    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray):
        # First, check if the maximum has changed. If so, update the maximum index. Then, update the
        # sum of priorities. This is done by subtracting the old priorities and adding the new ones
        # to the sum. Finally, update the priorities in the buffer.
        if np.max(priorities) > self._priorities[self._max_priority_idx]:
            self._max_priority_idx = indices[np.argmax(priorities)]
        self._sum_priorities_alpha -= np.sum(np.sqrt(self._priorities[indices]))
        self._priorities[indices] = priorities + 1e-10
        self._sum_priorities_alpha += np.sum(np.sqrt(self._priorities[indices]))

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
