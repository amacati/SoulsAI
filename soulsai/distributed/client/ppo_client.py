import logging
import time
from multiprocessing import Event

import gym

from soulsai.distributed.client.connector import PPOConnector

logger = logging.getLogger(__name__)


def ppo_client(config, tf_state_callback, tel_callback, encode_sample, encode_tel):
    logging.basicConfig(level=config.loglevel)
    logging.getLogger("soulsai").setLevel(config.loglevel)

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
            state = env.reset()
            done = False
            total_reward = 0.
            steps = 1
            while not done and not stop_flag.is_set():
                tfstate = tf_state_callback(state)
                action = con.agent.get_action(tfstate)
                next_state, reward, done, _ = env.step(action)
                total_reward += reward
                con.push_sample(con.agent.model_id, state, action, reward, next_state, done)
                state = next_state
                steps += 1
                ppo_steps += 1
                if config.step_delay:
                    time.sleep(config.step_delay)
                if ppo_steps == config.ppo_steps:
                    ppo_steps = 0
                    con.sync()  # Wait for the new model to download
            con.push_telemetry(*tel_callback(total_reward, steps, state))
    finally:
        env.close()
        con.close()
