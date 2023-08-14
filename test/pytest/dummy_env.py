from pathlib import Path
import json

import gymnasium


class IudexDummyEnv(gymnasium.Env):
    """Dummy environment that replays input data collected from a IudexEnv.

    The input data is saved in a json file and can be used to test the IudexObservationWrapper. To
    create the input data, run the script ``scripts/collect_reference_test_data.py``.
    """

    path = Path(__file__).parent / "data" / "iudex_env_reference_input.json"

    def __init__(self):
        super().__init__()
        with open(self.path, "r") as f:
            self.data = json.load(f)
        self._step = 0
        self._episode = -1
        self._done = True
        self._max_steps = len(self.data["observations"])

    def reset(self, seed: int | None = None, options: dict | None = None):
        assert self._step < self._max_steps, "No more data to replay!"
        self._done = False
        obs = self.data["observations"][self._step]
        info = self.data["infos"][self._step]
        self._step += 1
        self._episode += 1
        return obs, info

    def step(self, _: int):
        assert not self._done, "Environment requires reset!"
        assert self._step < self._max_steps, "No more data to replay!"
        obs = self.data["observations"][self._step]
        reward = self.data["rewards"][self._step]
        done = self.data["dones"][self._step]
        info = self.data["infos"][self._step]
        self._step += 1
        self._done = done
        return obs, reward, False, done, info
