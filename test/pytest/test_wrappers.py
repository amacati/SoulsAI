"""Test the observation wrapper for the Iudex environment."""

import json
from pathlib import Path

import numpy as np
from dummy_env import IudexDummyEnv

from soulsai.wrappers.iudex_wrappers import IudexObservationWrapper


def load_ground_truth_data() -> list[np.ndarray]:
    """Load ground truth data from the original transformation implementation.

    Returns:
        A list of reference output data.
    """
    with open(Path(__file__).parent / "data/iudex_env_reference_output.json", "r") as f:
        data = json.load(f)
    return [np.array(x) for x in data]


def test_iudex_observation_wrapper():
    """Compare the observations of the wrapped environment to the ground truth data.

    Load reference data from the original transformation implementation, create a wrapped dummy
    environment that replays the original inputs and compare the results.

    Note:
        Since the original input data is saved as a json file, we loose some precision. To
        compensate this we use ``np.allclose`` with a small atol to compare the observations.
    """
    gt_obs = load_ground_truth_data()

    env = IudexDummyEnv()
    env = IudexObservationWrapper(env)
    step = 0
    n_tests = 2
    for episode in range(n_tests):
        done = False
        obs, _ = env.reset()
        if not np.allclose(obs, gt_obs[step], atol=1e-7):
            raise AssertionError(
                f"Obs and ground truth do not match! Step {step}, episode {episode}"
            )
        step += 1
        while not done:
            # We disregard the action as the dummy env replays the action from the ground truth data
            obs, _, truncated, terminated, _ = env.step(0)
            done = terminated or truncated
            if not np.allclose(obs, gt_obs[step], atol=1e-7):
                raise AssertionError(
                    f"Obs and ground truth do not match! Step {step}, episode {episode}"
                )
            step += 1
