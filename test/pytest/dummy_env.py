"""Dummy environment for testing the environments without running soulsgym environments.

Since soulsgym requires Windows with a Dark Souls III installation, we cannot run tests on most
machines. The dummy environments replay input data collected from soulsgym environments and can be
used to test the observation wrappers.
"""

import json
from pathlib import Path

import gymnasium


class IudexDummyEnv(gymnasium.Env):
    """Dummy environment that replays input data collected from a IudexEnv.

    The input data is saved in a json file and can be used to test the IudexObservationWrapper. To
    create the input data, run the script ``scripts/collect_reference_test_data.py``.
    """

    path = Path(__file__).parent / "data" / "iudex_env_reference_input.json"
    step_size = 0.1

    def __init__(self):
        """Load the data from the json file and initialize environment helpers."""
        super().__init__()
        with open(self.path, "r") as f:
            self.data = json.load(f)
        self._step = 0
        self._episode = -1
        self._done = True
        self._max_steps = len(self.data["observations"])

    def reset(self, seed: int | None = None, options: dict | None = None) -> tuple[dict, dict]:
        """Reset the dummy environment.

        Ignores the seed and options and replays the return values from its dataset.

        Args:
            seed: Ignored seed.
            options: Ignored options.

        Returns:
            A tuple containing the observation and info.
        """
        assert self._step < self._max_steps, "No more data to replay!"
        self._done = False
        obs = self.data["observations"][self._step]
        info = self.data["infos"][self._step]
        self._step += 1
        self._episode += 1
        return obs, info

    def step(self, _: int) -> tuple[dict, float, bool, bool, dict]:
        """Step through the dummy environment.

        Ignores the action and replays the return values from its dataset.

        Args:
            _: Ignored action.

        Returns:
            A tuple containing the observation, reward, terminated, truncated and info.
        """
        assert not self._done, "Environment requires reset!"
        assert self._step < self._max_steps, "No more data to replay!"
        obs = self.data["observations"][self._step]
        reward = self.data["rewards"][self._step]
        done = self.data["dones"][self._step]
        info = self.data["infos"][self._step]
        self._step += 1
        self._done = done
        return obs, reward, False, done, info
