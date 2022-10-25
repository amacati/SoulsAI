import torch
import numpy as np

if __name__ == "__main__":
    # TODO: Check last advantage calculation
    # advantages = torch.zeros(11, dtype=torch.float64)
    # for t in reversed(range(n_steps)):
    #     td_error = rewards[t] + gamma * values[t+1] * (1.0 - dones[t]) - values[t]
    #     advantages[t] = td_error + gamma * gae_lambda * (1.0 - dones[t]) * advantages[t + 1]
    # returns_2 = (advantages + values)[:10]

    N = 8
    torch.random.manual_seed(0)
    mb_advantages = torch.rand(N) - 0.5
    ratio = torch.rand(N) * 2
