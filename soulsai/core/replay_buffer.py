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

    def __init__(self, maxlen, state_size):
        self.maxlen = maxlen
        self._idx = 0
        self._maxidx = -1
        self._b_s = torch.zeros((maxlen, state_size), dtype=torch.float32)
        self._b_a = torch.zeros((maxlen), dtype=torch.int64)
        self._b_r = torch.zeros((maxlen), dtype=torch.float32)
        self._b_sn = torch.zeros((maxlen, state_size), dtype=torch.float32)
        self._b_d = torch.zeros((maxlen), dtype=torch.float32)

    def append(self, experience: np.ndarray):
        self._b_s[self._idx, :] = torch.from_numpy(experience[0])
        self._b_a[self._idx] = experience[1]
        self._b_r[self._idx] = experience[2]
        self._b_sn[self._idx, :] = torch.from_numpy(experience[3])
        self._b_d[self._idx] = experience[4]
        self._idx = (self._idx + 1) % self.maxlen
        self._maxidx = min(self._maxidx + 1, self.maxlen - 1)

    def clear(self):
        self._idx = 0
        self._maxidx = 0

    def __len__(self):
        return self._maxidx + 1

    @property
    def filled(self):
        return self._maxidx + 1 == self.maxlen

    def sample_batch(self, n):
        if n > self._maxidx + 1:
            raise RuntimeError("Asked to sample more elements than available in buffer")
        i = np.random.choice(self._maxidx + 1, n, replace=False)
        return self._b_s[i, :], self._b_a[i], self._b_r[i], self._b_sn[i, :], self._b_d[i]

    def save(self, path):
        save_dict = {"_b_s": self._b_s, "_b_a": self._b_a, "_b_r": self._b_r, "_b_sn": self._b_sn,
                     "_b_d": self._b_d, "_idx": self._idx, "_maxidx": self._maxidx}
        torch.save(save_dict, path)

    def load(self, path):
        save_dict = torch.load(path)
        for att in ("_b_s", "_b_a", "_b_r", "_b_sn", "_b_d", "_idx", "_maxidx"):
            setattr(self, att, save_dict[att])
        self.maxlen = self._b_s.shape[0]


class MultistepEpisodeBuffer:

    def __init__(self, gamma):
        self.gamma = gamma
        self.buffer = []

    def append(self, experience):
        self.buffer.append(experience)

    def __len__(self):
        return len(self.buffer)

    def compute_multistep_returns(self, n_steps):
        np_gamma = np.flip(np.array([self.gamma**t for t in range(n_steps)]))
        rewards = np.array([exp[2] for exp in self.buffer])
        multistep_rewards = np.convolve(np_gamma, rewards)[n_steps-1:]
        for idx in range(len(self.buffer)-n_steps):
            self.buffer[idx][2] = multistep_rewards[idx]
            self.buffer[idx][3] = self.buffer[idx+n_steps][3]
            self.buffer[idx][4] = self.buffer[idx+n_steps][4]
        self.buffer = self.buffer[:-n_steps]

    def clear(self):
        self.buffer = []
