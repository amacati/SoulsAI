"""Evaluate the performance of the replay buffer implementation."""
import timeit
from soulsai.core.replay_buffer import ReplayBuffer
import torch
from tensordict import TensorDict

buffer_size = 100_000

rb = ReplayBuffer(buffer_size, device="cpu")
sample = {
    'obs': torch.randn(1, 10, 10),
    'action': torch.tensor([[1]]),
    'reward': torch.randn(1, 1),
    'nextObs': torch.randn(1, 10, 10),
    'terminated': torch.randn(1, 1),
    'truncated': torch.randn(1, 1)
}
sample_td = TensorDict(sample, batch_size=1, device='cpu')
rb.append(sample_td)  # Allocate buffers


def main():
    """Benchmark the `append`, `sample_batch` and `sample_batches` methods of the replay buffer."""
    repetitions = 1_000
    setup = ("from soulsai.core.replay_buffer import ReplayBuffer\n"
             "import torch\n"
             "from tensordict import TensorDict\n"
             "from __main__ import rb\n"
             "sample = {"
             "'obs': torch.randn(1, 10, 10),"
             "'action': torch.tensor([[1]]),"
             "'reward': torch.randn(1, 1),"
             "'nextObs': torch.randn(1, 10, 10),"
             "'terminated': torch.randn(1, 1),"
             "'truncated': torch.randn(1, 1)}\n"
             "sample_td = TensorDict(sample, batch_size=1, device='cpu')")
    stmt = "rb.append(sample_td)"
    time_1 = timeit.timeit(stmt, setup=setup, number=repetitions) / repetitions
    print(f"Append took an average of {time_1:.2e} seconds per call.")

    stmt = "rb.sample_batch(64)"
    time_3 = timeit.timeit(stmt, setup=setup, number=repetitions) / repetitions
    print(f"Sample took an average of {time_3:.2e} seconds per call.")

    stmt = "rb.sample_batches(64, 64)"
    time_5 = timeit.timeit(stmt, setup=setup, number=repetitions) / repetitions
    print(f"Batch sample took an average of {time_5:.2e} seconds per call.")


if __name__ == "__main__":
    main()
