from collections import deque
import pickle

import numpy as np
import torch


class ExperienceReplayBuffer:

    def __init__(self, maxlen=1000):
        self.maxlen = maxlen
        self.buffer = deque(maxlen=maxlen)

    def append(self, experience):
        self.buffer.append(experience)

    def __len__(self):
        return len(self.buffer)

    @property
    def filled(self):
        return len(self.buffer) == self.maxlen

    def sample_batch(self, n):
        if n > len(self.buffer):
            raise RuntimeError("Asked to sample more elements than available in buffer")
        indices = np.random.choice(len(self.buffer), n, replace=False)
        batch = [self.buffer[i] for i in indices]
        return zip(*batch)

    def clear(self):
        self.buffer.clear()

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(self.buffer, f)

    def load(self, path):
        with open(path, "rb") as f:
            self.buffer = pickle.load(f)
        self.maxlen = len(self.buffer)


class PerformanceBuffer:

    def __init__(self, maxlen, state_size, n_actions, action_masking=False):
        self.maxlen = maxlen
        self._idx = 0
        self._maxidx = -1
        self._b_s = np.zeros((maxlen, state_size))
        self._b_a = np.zeros(maxlen, dtype=np.int64)
        self._b_am = np.zeros((maxlen, n_actions))
        self._b_r = np.zeros(maxlen)
        self._b_sn = np.zeros((maxlen, state_size))
        self._b_d = np.zeros(maxlen)
        self._action_masking = action_masking

    def append(self, experience):
        self._b_s[self._idx] = experience[0]
        self._b_a[self._idx] = experience[1]
        self._b_r[self._idx] = experience[2]
        self._b_sn[self._idx] = experience[3]
        self._b_d[self._idx] = experience[4]
        if self._action_masking:
            self._b_am[self._idx] = 0
            self._b_am[self._idx, experience[5]["allowed_actions"]] = 1
        self._idx = (self._idx + 1) % self.maxlen
        self._maxidx = min(self._maxidx + 1, self.maxlen - 1)

    def clear(self):
        self._idx = 0
        self._maxidx = -1

    def __len__(self):
        return self._maxidx + 1

    @property
    def filled(self):
        return self._maxidx + 1 == self.maxlen

    def sample_batch(self, n):
        if n > self._maxidx + 1:
            raise RuntimeError("Asked to sample more elements than available in buffer")
        i = np.random.choice(self._maxidx + 1, n, replace=False)
        if self._action_masking:
            return self._b_s[i], self._b_a[i], self._b_r[i], self._b_sn[i], self._b_d[i], self._b_am[i]  # noqa: E501
        return self._b_s[i], self._b_a[i], self._b_r[i], self._b_sn[i], self._b_d[i]

    def sample_batches(self, nsamples, nbatches):
        if nsamples > self._maxidx + 1:
            raise RuntimeError("Asked to sample more elements than available in buffer")
        # If the buffer contains more samples than requested in total, indices are chosen such that
        # no sample is sampled twice across all batches. If more total samples are requested than
        # available in the buffer, resort to random independent indices in each batch
        if nsamples * nbatches <= self._maxidx + 1:
            unique_indices = np.random.choice(self._maxidx + 1, nsamples * nbatches, replace=False)
            indices = np.split(unique_indices, nbatches)
        else:
            indices = [np.random.choice(self._maxidx + 1, nsamples, replace=False)
                       for _ in range(nbatches)]
        if self._action_masking:
            batches = [[self._b_s[i], self._b_a[i], self._b_r[i], self._b_sn[i], self._b_d[i],
                        self._b_am[i]] for i in indices]
        else:
            batches = [[self._b_s[i], self._b_a[i], self._b_r[i], self._b_sn[i], self._b_d[i]]
                       for i in indices]
        return batches

    def save(self, path):
        save_dict = {"_b_s": self._b_s, "_b_a": self._b_a, "_b_r": self._b_r, "_b_sn": self._b_sn,
                     "_b_d": self._b_d, "_idx": self._idx, "_maxidx": self._maxidx}
        if self._action_masking:
            save_dict["_b_am"] = self._b_am
        torch.save(save_dict, path)

    def load(self, path):
        save_dict = torch.load(path)
        for att in ("_b_s", "_b_a", "_b_r", "_b_sn", "_b_d", "_idx", "_maxidx"):
            setattr(self, att, save_dict[att])
        if self._action_masking:
            self._b_am = save_dict["_b_am"]
        self.maxlen = self._b_s.shape[0]


class TrajectoryBuffer:

    def __init__(self, n_trajectories, n_samples, n_states, n_actions, categorical=True):
        self.n_trajectories = n_trajectories
        self.n_samples = n_samples
        self.n_batch_samples = n_trajectories * n_samples
        self.n_states = n_states
        self.categorical = categorical
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

    def append(self, sample):
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
        self._complete_flags[:] = False
        self.advantages[:] = torch.nan  # Make sure to not accidentally use old advantages

    @property
    def buffer_complete(self):
        return np.all(self._complete_flags)

    def compute_advantages_and_values(self, agent, gamma, gae_lambda):
        self.values = agent.get_values(self.states, requires_grad=False)
        self._end_values = agent.get_values(self._end_states, requires_grad=False)
        for trajectory_id in range(self.n_trajectories):
            last_advantage = 0
            for step_id in reversed(range(self.n_samples)):
                idx = trajectory_id * self.n_samples + step_id
                if step_id == self.n_samples - 1:  # Terminal sample
                    not_done = (1. - self._end_dones[trajectory_id])
                    next_value = self._end_values[trajectory_id]
                else:
                    not_done = (1. - self.dones[idx])
                    next_value = self.values[idx + 1]
                td_error = self.rewards[idx] + gamma * next_value * not_done - self.values[idx]
                last_advantage = td_error + gamma * gae_lambda * not_done * last_advantage
                self.advantages[idx] = last_advantage
