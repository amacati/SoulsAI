"""The replay buffer module offers performant implementations of replay buffers for DQN and PPO."""

from __future__ import annotations

import random
from abc import ABC, abstractmethod, abstractproperty
from typing import TYPE_CHECKING, Callable

import torch
from tensordict import TensorDict

from soulsai.utils import module_type_from_string

if TYPE_CHECKING:
    from pathlib import Path

buffer_cls: Callable[[str], type[AbstractBuffer]] = module_type_from_string(__name__)


class AbstractBuffer(ABC):
    """Abstract replay buffer class."""

    def __init__(self):
        """Initialize the buffer."""
        super().__init__()

    @abstractproperty
    def size(self) -> int:
        """Get the buffer size.

        Returns:
            The buffer size.
        """

    @abstractmethod
    def append(self, sample: dict):
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
    def sample_batch(self, batch_size: int) -> TensorDict:
        """Sample a single batch from the buffer.

        Args:
            batch_size: Number of samples in the batch.

        Returns:
            The sampled batch.
        """

    @abstractmethod
    def sample_batches(self, batch_size: int, nbatches: int) -> TensorDict:
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
    already filled with experience. If reproducible sampling is required, the seeds of random and
    torch have to be set before using the buffer.
    """

    def __init__(self, max_size: int, device: torch.device = torch.device("cpu")):
        """Create the buffer tensor dict and set the index to 0.

        Args:
            max_size: Maximum buffer capacity.
            device: Buffer device.
        """
        super().__init__()
        self.max_size = max_size
        self.device = device
        self.buffer: TensorDict[torch.Tensor] = TensorDict({}, batch_size=max_size, device=device)
        # Helper indices to implement a ring buffer
        self._idx = 0
        self._maxidx = -1

    @property
    def size(self) -> int:
        """Get the buffer size.

        Returns:
            The buffer size.
        """
        return self.max_size

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
                self.buffer[key] = torch.zeros(
                    (self.max_size, *sample[key].shape[1:]),
                    dtype=sample[key].dtype,
                    device=self.device,
                )
            elif isinstance(sample[key], TensorDict):
                self.buffer[key] = TensorDict({}, batch_size=self.max_size, device=self.device)
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
        return self.buffer[torch.tensor(random.sample(range(self._maxidx + 1), batch_size))]

    def sample_batches(self, batch_size: int, nbatches: int) -> TensorDict[torch.Tensor]:
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
            i = torch.tensor(random.sample(range(self._maxidx + 1), nsamples))
        else:
            i = torch.randint(0, self._maxidx + 1, (nsamples,), device=self.device)
        return self.buffer[i].reshape(nbatches, batch_size)


class PrioritizedReplayBuffer(ReplayBuffer):
    """Implementation of a prioritized replay buffer.

    Buffers for samples are allocated on receiving the first sample. An internal index keeps track
    of the current size of the buffer and enables to only sample from the parts of the buffers
    already filled with experience. If reproducible sampling is required, the seeds of random and
    torch have to be set before using the buffer.

    We fix alpha to 0.5 and use sqrt instead of pow(alpha). See e.g. Dopamine implementation at
    https://github.com/google/dopamine/blob/a6f414ca01a81e933359a4922965178a40e0f38a/dopamine/jax/agents/quantile/quantile_agent.py#L262
    """

    def __init__(
        self, max_size: int, beta: float = 0.5, device: torch.device = torch.device("cpu")
    ):
        """Preallocate the buffer arrays and set the index to 0.

        Args:
            max_size: Maximum buffer capacity.
            seed: Random seed for reproducible sampling.
            beta: Weight correction exponent. Controls how much bias correction is used.
            device: Buffer device
        """
        super().__init__(max_size=max_size, device=device)
        # Initialize helper variables for tracking the priorities for each sample
        self.buffer["__priority__"] = torch.zeros(
            max_size, dtype=torch.float32, requires_grad=False, device=device
        )
        self.buffer["__priority_sqrt__"] = torch.zeros(
            max_size, dtype=torch.float32, requires_grad=False, device=device
        )
        self._sum_priorities_alpha = torch.zeros(1, dtype=torch.float64, device=device)
        self._max_priority_idx = 0
        self.beta = torch.tensor(beta, dtype=torch.float64, device=device)

    def append(self, sample: TensorDict[torch.Tensor]):
        """Append a sample to the buffer.

        Note:
            The sample must not contain the '__priority__' key.

        Args:
            sample: Sample dictionary containing the observation, action, reward etc.
        """
        num_samples = sample.batch_size[0]
        assert num_samples < self.max_size, "Sample size must be smaller than the buffer"
        self._allocate_buffers(sample)
        # Update the sum of priorities (to the power of alpha) and the maximum priority index. We
        # keep track of the sum of priorities to avoid having to recompute it every time we sample
        # from the buffer.
        idx = torch.arange(self._idx, self._idx + num_samples) % self.max_size
        self.buffer[idx] = sample
        self._sum_priorities_alpha -= self.buffer["__priority__"][self._idx].sqrt().sum()
        # Check if the maximum index is about to be overwritten. If it is, we first need to
        # recompute the new maximum index and priority. In practice, this does not happen often, as
        # older samples should generally have a lower TD error and therefore lower priority.
        if (idx == self._max_priority_idx).any():
            self.buffer["__priority__"][idx] = 0.0  # Set to 0 to remove from max calculation
            self._max_priority_idx = torch.argmax(self.buffer["__priority__"])
        # Set the priortiy of the new sample to the current maximum priority and update the sum of
        # priorities
        priority = 1.0 if self._maxidx < 0 else self.buffer["__priority__"][self._max_priority_idx]
        self.buffer["__priority__"][idx] = priority + 1e-10
        # Cache the square root of the priority for faster sampling
        self.buffer["__priority_sqrt__"][idx] = self.buffer["__priority__"][idx].sqrt()
        self._sum_priorities_alpha += self.buffer["__priority_sqrt__"][idx].sum()
        # Update the internal index and the maximum index
        self._idx = (self._idx + num_samples) % self.max_size
        self._maxidx = min(self._maxidx + num_samples, self.max_size - 1)

    def sample_batch(self, batch_size: int) -> TensorDict[torch.Tensor]:
        """Sample a single batch from the buffer.

        Args:
            batch_size: Number of samples in the batch.

        Returns:
            The sampled batch. Index and weight of the samples can be accessed with the '__idx__'
                and '__weight__' keys.

        Raises:
            RuntimeError: Asked to sample more samples than currently available.
        """
        if batch_size > self._maxidx + 1:
            raise RuntimeError("Asked to sample more elements than available in buffer")
        prob = self.buffer["__priority_sqrt__"][: self._maxidx + 1] / self._sum_priorities_alpha
        idx = torch.multinomial(prob, batch_size)
        batch = self.buffer[idx]
        # Add information about the weights and the indices to the batch
        weights = ((self._maxidx + 1) * prob[idx]).pow(-self.beta)
        normalized_weights = weights / weights.max()
        batch["__idx__"] = idx
        batch["__weight__"] = normalized_weights
        return batch

    def sample_batches(self, batch_size: int, nbatches: int) -> TensorDict[torch.Tensor]:
        """Sample multiple batches from the buffer.

        Args:
            batch_size: Number of samples per batch.
            nbatches: Number of batches.

        Returns:
            The sampled batches including the weights and the indices of the samples.

        Raises:
            RuntimeError: Asked to sample more samples per batch than currently available.
        """
        return self.sample_batch(batch_size=batch_size * nbatches).reshape(nbatches, batch_size)

    def update_priorities(self, batch: TensorDict[torch.Tensor]):
        """Update the priorities of the samples at the specified indices.

        Note:
            No samples must be added or removed from the buffer between the sampling and the
            priority update.

        Args:
            batch: Batch of samples with updated priorities. The batch must contain the keys
                '__idx__' and '__priority__'.
        """
        # First, check if the maximum has changed. If so, update the maximum index. Then, update the
        # sum of priorities by subtracting the old priorities and adding the new ones to the sum.
        # Finally, update the priorities in the buffer.
        idx = batch["__idx__"].to(self.device)
        max_priority, max_idx = torch.max(batch["__priority__"], dim=-1)
        assert max_priority.shape == tuple(), f"Invalid shape {max_priority.shape}"
        if max_priority > self.buffer["__priority__"][self._max_priority_idx]:
            self._max_priority_idx = idx[max_idx]
        self._sum_priorities_alpha -= self.buffer["__priority_sqrt__"][idx].sum().item()
        self.buffer["__priority__"][idx] = batch["__priority__"].to(self.device) + 1e-10
        self.buffer["__priority_sqrt__"][idx] = self.buffer["__priority__"][idx].sqrt()
        self._sum_priorities_alpha += self.buffer["__priority_sqrt__"][idx].sum().item()


class TrajectoryBuffer:
    """Experience buffer to hold samples of multiple trajectories while preserving their order.

    The buffer has a fixed amount of trajectories with a fixed amount of samples. Each sample has a
    unique trajectory ID and a step ID. The buffer is full when the samples from all trajectories
    for all steps have been appended. Samples can be appended out of order since the IDs are used to
    sort them into the correct order.

    Note:
        This buffer is designed for categorical actions PPO only!
    """

    def __init__(
        self, n_trajectories: int, n_samples: int, device: torch.device = torch.device("cpu")
    ):
        """Preallocate the sample tensors and completion flags.

        Args:
            n_trajectories: Number of parallel trajectories from parallel environments used.
            n_samples: Number of samples for each trajectory.
            obs_shape: Observation shape.
            device: Buffer device.
        """
        self.n_trajectories = n_trajectories
        self.n_samples = n_samples
        self.n_batch_samples = n_trajectories * n_samples
        self.device = device
        self.buffer = TensorDict({}, batch_size=(n_trajectories, n_samples), device=device)
        # For the advantage calculation, we need the next observation after the final sample.
        self.final_buffer = TensorDict({}, batch_size=n_trajectories, device=device)

        self.values = torch.zeros((n_trajectories * n_samples, 1), dtype=torch.float32)
        self.advantages = torch.zeros((n_trajectories, n_samples), dtype=torch.float32)
        # Create a tensor that tracks which samples have already been added for fast checking
        self._complete_flags = torch.zeros((n_trajectories, n_samples + 1), dtype=torch.bool)

    def append(self, sample: dict):
        """Append a PPO sample to the buffer.

        Also sets the complete flag for the received sample.

        Args:
            sample: PPO sample consisting of the observation, the chosen action, the action
            probability, the reward, the terminated flag, the trajectory ID and the step ID.
        """
        trajectory_id, step_id = sample["client_id"], sample["step_id"]
        assert trajectory_id < self.n_trajectories and step_id <= self.n_samples
        if step_id != self.n_samples:  # Non-terminal sample
            self._allocate_buffers(self.buffer, sample)
            self.buffer[trajectory_id, step_id] = sample
        else:
            self._allocate_buffers(self.final_buffer, sample)
            self.final_buffer[trajectory_id] = sample
        self._complete_flags[trajectory_id, step_id] = True

    def _allocate_buffers(self, buffer: TensorDict, sample: TensorDict[torch.Tensor]):
        # Check if there are unknown keys in the sample and allocate buffers for them if necessary
        for key in set(sample.keys()).difference(set(buffer.keys())):
            if isinstance(sample[key], torch.Tensor):
                size = (*buffer.batch_size, *sample[key].shape[1:])
                buffer[key] = torch.zeros(*size, dtype=sample[key].dtype, device=self.device)
            elif isinstance(sample[key], TensorDict):
                buffer[key] = TensorDict(
                    sample[key], batch_size=buffer.batch_size, device=self.device
                )
            else:
                raise ValueError(f"Unknown sample type for key {key}")

    def clear(self):
        """Clear the buffer complete flags and reset the advantage values."""
        self._complete_flags[:] = False
        self.advantages[:] = torch.nan  # Make sure to not accidentally use old advantages

    @property
    def buffer_complete(self) -> bool:
        """Flag to check if all required samples have been added to the buffer."""
        return torch.all(self._complete_flags)
