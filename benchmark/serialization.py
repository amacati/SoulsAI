import timeit

setup = """import torch\n
import numpy as np
from tensordict import TensorDict

from soulsai.distributed.common.serialization import serialize


sample = {
    "obs": np.zeros((12, 90, 90)),
    "action": 1,
    "reward": 1.,
    "nextObs": np.zeros((12, 90, 90)),
    "terminated": False,
    "truncated": False,
    "info": {"allowed_actions": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]},
    "modelId": "nope",
}
sample_td = TensorDict(
    {
        k: torch.tensor(v).unsqueeze(0) for k, v in sample.items() if k not in ("info", "modelId")
    },
    batch_size=1,
    device="cpu")
sample_td["info"] = TensorDict(
    {"allowed_actions": torch.tensor(sample["info"]["allowed_actions"]).unsqueeze(0)},
    batch_size=1,
    device="cpu")
sample_td["modelId"] = torch.tensor([0])
"""


def main():
    repetitions = 1_000
    stmt = f"serialize(sample)"
    time_2 = timeit.timeit(stmt, setup=setup, number=repetitions) / repetitions
    print(f"Serialize took an average of {time_2:.2e} seconds per call.")


if __name__ == "__main__":
    main()
