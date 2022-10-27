import logging
import time
from multiprocessing import Event

import gym

from soulsai.distributed.client.connector import PPOConnector

logger = logging.getLogger(__name__)


def ppo_client(config, tf_state_callback, tel_callback, encode_sample, encode_tel,
               episode_end_callback=None):
    logging.basicConfig(level=config.loglevel)
    logging.getLogger("soulsai").setLevel(config.loglevel)
    logger.info("Launching PPO client")

    stop_flag = Event()
    if config.enable_interrupt:
        import keyboard  # Keyboard should not be imported in Docker during testing

        def exit_callback():
            stop_flag.set()

        keyboard.add_hotkey("enter", exit_callback)
        logger.info("Press 'Enter' to end training")

    env = gym.make(config.env)
    con = PPOConnector(config, encode_sample, encode_tel)
    con.sync()  # Wait for the new model to download

    logger.info("Client node running")
    try:
        episode_id = 0
        ppo_steps = 0
        while not stop_flag.is_set() and episode_id != config.max_episodes:
            episode_id += 1
            state = tf_state_callback(env.reset())
            done = False
            total_reward = 0.
            steps = 1
            while not done and not stop_flag.is_set():
                action, prob = con.agent.get_action(state)
                next_state, reward, next_done, _ = env.step(action)
                next_state = tf_state_callback(next_state)
                total_reward += reward
                con.push_sample(con.agent.model_id, ppo_steps, [state, action, prob, reward, done])
                logger.debug(f"Pushed sample {ppo_steps} for model {con.agent.model_id}")
                state = next_state
                done = next_done
                steps += 1
                ppo_steps += 1
                if config.step_delay:
                    time.sleep(config.step_delay)
                if ppo_steps == config.ppo.n_steps:
                    con.push_sample(con.agent.model_id, ppo_steps, [next_state, 0, [0], 0, done])
                    ppo_steps = 0
                    con.sync(config.ppo.client_sync_timeout)  # Wait for the new model
            con.push_telemetry(*tel_callback(total_reward, steps, state))
            if episode_end_callback is not None:
                episode_end_callback()
    finally:
        env.close()
        con.close()
