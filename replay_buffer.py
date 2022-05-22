from collections import deque
import pickle

import numpy as np


class ExperienceReplayBuffer:

    def __init__(self, maximum_length=1000):
        self.maximum_length = maximum_length
        self.buffer = deque(maxlen=maximum_length)

    def append(self, experience):
        self.buffer.append(experience)

    def __len__(self):
        return len(self.buffer)

    def sample_batch(self, n):
        if n > len(self.buffer):
            raise RuntimeError("Asked to sample more elements than available in buffer")
        indices = np.random.choice(len(self.buffer), n, replace=False)
        batch = [self.buffer[i] for i in indices]
        return zip(*batch)

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(self.buffer, f)

    def load(self, path):
        with open(path, "rb") as f:
            self.buffer = pickle.load(f)


class ImportanceReplayBuffer:

    def __init__(self, maximum_length=1000):
        self.maximum_length = maximum_length
        self.buffer = deque(maxlen=maximum_length)
        self.p = deque(maxlen=maximum_length)

    def append(self, experience):
        self.buffer.append(experience)

    def __len__(self):
        return len(self.buffer)

    def sample_batch(self, n):
        if n > len(self.buffer):
            raise RuntimeError("Asked to sample more elements than available in buffer")
        indices = np.random.choice(np.arange(len(self.buffer)), size=n, replace=False, p=self.p)
        batch = [self.buffer[i] for i in indices]
        return map(np.array, zip(*batch))

    def get_all_samples(self):
        return map(np.array, zip(*self.buffer))


class EpisodeBuffer:

    def __init__(self):
        self.buffer = []

    def append(self, experience):
        self.buffer.append(experience)

    def __len__(self):
        return len(self.buffer)

    def sample(self):
        return map(np.array, zip(*self.buffer))

    def clear(self):
        self.buffer = []


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
