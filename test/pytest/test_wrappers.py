import numpy as np

from dummy_env import IudexDummyEnv


def load_ground_truth_data():
    ...


def test_iudex_observation_wrapper():
    from soulsai.wrappers.iudex_wrappers import IudexObservationWrapper
    gt_obs, gt_reward, gt_info = load_ground_truth_data()

    env = IudexDummyEnv()
    env = IudexObservationWrapper(env)
    for episode in range(10):
        done, step = False, 0
        obs, info = env.reset()
        assert np.all(obs == gt_obs[episode, step])
        assert info == gt_info[episode, step]
        while not done:
            # We disregard the action as the dummy env replays the action from the ground truth data
            step += 1
            obs, reward, truncated, terminated, info = env.step(0)
            done = terminated or truncated
            assert np.all(obs == gt_obs[episode, step])
            assert reward == gt_reward[episode, step]
            assert info == gt_info[episode, step]
