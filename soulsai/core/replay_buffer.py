"""The replay buffer module offers performant implementations of replay buffers for DQN and PPO."""
from pathlib import Path
from typing import Tuple, List

import numpy as np
import torch

from soulsai.core.agent import PPOAgent

PPOSample = Tuple[np.ndarray, int, float, float, bool, int, int]
DQNSample = Tuple[np.ndarray, int, float, np.ndarray, bool]
DQNSampleWithActionMask = Tuple[np.ndarray, int, float, np.ndarray, bool, dict]


class PerformanceBuffer:
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

    def append(self, sample: DQNSample | DQNSampleWithActionMask):
        """Append a sample to the buffer.

        Args:
            sample: DQN sample containing (in that order) the state, the action, the reward, the
                next state, and the done flag. If action masking is used, also contains the env info
                dict.
        """
        self._b_s[self._idx] = sample[0]
        self._b_a[self._idx] = sample[1]
        self._b_r[self._idx] = sample[2]
        self._b_sn[self._idx] = sample[3]
        self._b_d[self._idx] = sample[4]
        if self._action_masking:
            self._b_am[self._idx] = 0
            self._b_am[self._idx, sample[5]["allowed_actions"]] = 1
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
            return [self._b_s[i], self._b_a[i], self._b_r[i], self._b_sn[i], self._b_d[i], self._b_am[i]]  # noqa: E501
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
            indices = [np.random.choice(self._maxidx + 1, batch_size, replace=False)
                       for _ in range(nbatches)]
        if self._action_masking:
            batches = [[self._b_s[i], self._b_a[i], self._b_r[i], self._b_sn[i], self._b_d[i],
                        self._b_am[i]] for i in indices]
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
        save_dict = {"_b_s": self._b_s, "_b_a": self._b_a, "_b_r": self._b_r, "_b_sn": self._b_sn,
                     "_b_d": self._b_d, "_idx": self._idx, "_maxidx": self._maxidx}
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

    def append(self, sample: PPOSample):
        """Append a PPO sample to the buffer.

        Also sets the complete flag for the received sample.

        Args:
            sample: PPO sample consisting of (in that order) the state, the chosen action, the
                action probability, the reward, the done flag, the trajectory ID and the step ID.
        """
        trajectory_id, step_id = sample[5], sample[6]
        assert trajectory_id < self.n_trajectories and step_id <= self.n_samples
        if step_id != self.n_samples:  # Non-terminal sample
            idx = trajectory_id * self.n_samples + step_id
            self.states[idx, :] = torch.from_numpy(sample[0])
            self.actions[idx] = sample[1]
            self.probs[idx] = sample[2]
            self.rewards[idx] = sample[3]
            self.dones[idx] = sample[4]
        else:
            self._end_states[trajectory_id] = torch.from_numpy(sample[0])
            self._end_dones[trajectory_id] = sample[4]
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
