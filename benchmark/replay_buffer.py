import timeit
from soulsai.core.replay_buffer import ReplayBuffer, DynamicReplayBuffer
import torch
from tensordict import TensorDict

obs, act, rew, next_obs, done = 0, 1, 2, 3, 4
sample = {
    "obs": torch.randn(10, 10),
    "action": torch.tensor([1]),
    "reward": torch.randn(1),
    "nextObs": torch.randn(10, 10),
    "terminated": torch.randn(1),
    "truncated": torch.randn(1)
}
sample2 = TensorDict({k: v.unsqueeze(0) for k, v in sample.items()}, batch_size=1, device="cpu")

buffer_size = 100_000
rb = ReplayBuffer(buffer_size, (10, 10), 10)

dynamic_rb = DynamicReplayBuffer(buffer_size)
dynamic_rb.append(sample2)


# Implementation 1
def append_1(sample, sample2):
    for _ in range(100):
        rb.append(sample)


# Implementation 2
def append_2(sample, sample2):
    for _ in range(100):
        dynamic_rb.append(sample2)


def sample_1(*args):
    for _ in range(100):
        rb.sample_batch(10)


def sample_1b(*args):
    for _ in range(100):
        rb.sample_batches(10, 10)


def sample_2(*args):
    for _ in range(100):
        dynamic_rb.sample_batch(10)


def sample_2b(*args):
    for _ in range(100):
        dynamic_rb.sample_batches(10, 10)


# Benchmark function
def benchmark(func, repetitions=10):
    setup_code = (f"from __main__ import {func}, sample, sample2\n"
                  "from soulsai.core.replay_buffer import ReplayBuffer\n"
                  "")
    stmt = f"{func}(sample, sample2)"
    time_taken = timeit.timeit(stmt, setup=setup_code, number=repetitions)
    return time_taken / repetitions  # Average time per function call


def main():
    # Set the limit for the sum of squares calculation
    limit = 1000

    # Benchmark the first implementation
    time_1 = benchmark("append_1", limit)
    print(f"Append 1 took an average of {time_1:.6f} seconds per call.")

    # Benchmark the second implementation
    time_2 = benchmark("append_2", limit)
    print(f"Append 2 took an average of {time_2:.6f} seconds per call.")

    time_3 = benchmark("sample_1", limit)
    print(f"Sample 1 took an average of {time_3:.6f} seconds per call.")

    time_4 = benchmark("sample_2", limit)
    print(f"Sample 2 took an average of {time_4:.6f} seconds per call.")

    time_5 = benchmark("sample_1b", limit)
    print(f"Sample 1b took an average of {time_5:.6f} seconds per call.")

    time_6 = benchmark("sample_2b", limit)
    print(f"Sample 2b took an average of {time_6:.6f} seconds per call.")

    # Compare results
    if time_1 < time_2:
        print("Implementation 1 is faster.")
    elif time_1 > time_2:
        print("Implementation 2 is faster.")
    else:
        print("Both implementations have similar performance.")


if __name__ == "__main__":
    main()
