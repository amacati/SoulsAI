import time

import numpy as np

from soulsgym.core.game_state import GameState
from soulsai.core.replay_buffer import ExperienceReplayBuffer, PerformanceBuffer
from soulsai.core.utils import gamestate2np
from soulsai.core.agent import DQNAgent


def main():
    nsamples = 100
    batch_size = 64
    nepochs = 50
    nruns = 100
    size_s = 72
    dummy_buffer = ExperienceReplayBuffer(nsamples)
    perf_buffer = PerformanceBuffer(nsamples+1, size_s)
    agent = DQNAgent(72, 20, 1e-3, 0.99, 5., 200)
    for _ in range(nsamples):
        gs = GameState(player_max_hp=1, player_max_sp=1, boss_max_hp=1, player_animation="Idle",
                       boss_animation="Idle")
        sample = (gs, np.random.randint(0, 20), np.random.random(), gs, False)
        dummy_buffer.append(sample)
        ngs = gamestate2np(gs)
        sample = (ngs, np.random.randint(0, 20), np.random.random(), ngs, False)
        perf_buffer.append(sample)
    t0 = time.time()
    perf_buffer.save("perf_buffer.pt")
    perf_buffer.load("perf_buffer.pt")
    tdata = 0
    ttrain = 0
    for _ in range(nruns):
        for _ in range(nepochs):
            t_0 = time.time()
            states, actions, rewards, next_states, dones = perf_buffer.sample_batch(batch_size)
            t_1 = time.time()
            agent.train(states, actions, rewards, next_states, dones)
            t_2 = time.time()
            tdata += t_1 - t_0
            ttrain += t_2 - t_1
    t1 = time.time()
    print(f"Total: {(t1-t0)/nruns:.3e}s, Data: {tdata/nruns:.3e}s, Training: {ttrain/nruns:.3e}s")


if __name__ == "__main__":
    main()
