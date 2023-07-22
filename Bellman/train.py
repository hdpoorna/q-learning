"""
py311
hdpoorna
"""

# import packages
import os
from tqdm import tqdm
from datetime import datetime
import gymnasium as gym
import numpy as np
from helpers import config
from helpers.q_table_helper import *

# make the env
env = gym.make("MountainCar-v0")

# set constants
config.OBS_HIGHS = env.observation_space.high
config.OBS_LOWS = env.observation_space.low
config.NUM_ACTIONS = env.action_space.n
config.ALL_ACTIONS = np.array(range(config.NUM_ACTIONS))
config.GOAL_POSITION = env.goal_position

# initialize q table
q_table = init_q_table(env, init_dist="normal")
# q_table = init_q_table(env, init_dist="uniform")
# q_table = load_q_table("q_tables/q_table_final.npy")
config.NUM_BUCKETS = list(q_table.shape[:-1])
config.BUCKET_SIZES = (config.OBS_HIGHS - config.OBS_LOWS)/config.NUM_BUCKETS


# create directories to save qtable and results
now_utc = datetime.utcnow()
now_str = now_utc.strftime("%Y-%m-%d-%H-%M-%S")
config.QTABLE_ID = "{}-{}".format(config.QTABLE_ID, now_str)
QTABLES_DIR = os.path.join("q_tables", config.QTABLE_ID)
RESULTS_DIR = os.path.join("results", config.QTABLE_ID)
make_dir(QTABLES_DIR)
make_dir(RESULTS_DIR)


# write configs to txt
write_to_txt(config.QTABLE_ID)


def select_action(state_bucket, episode):
    if config.EXPLORATION > 0.0:
        if config.START_EXPLORING <= episode <= config.END_EXPLORING:
            if np.random.random() <= config.EXPLORATION:
                action = np.random.choice(config.ALL_ACTIONS, 1)[0]
            else:
                action = np.argmax(q_table[state_bucket])
        else:
            action = np.argmax(q_table[state_bucket])
    else:
        action = np.argmax(q_table[state_bucket])
    return action


# collect rewards and exploration rates to graph
rewards = np.zeros(config.EPISODES, dtype=np.float32)
explorations = np.zeros(config.EPISODES, dtype=np.float32)


print("training starting!")
for episode in tqdm(range(config.EPISODES), ascii=True, unit="episodes"):
    # get initial state
    state_bucket = get_state_bucket(env.reset()[0], config.OBS_LOWS, config.BUCKET_SIZES)

    terminated = False      # goal achieved
    truncated = False       # timed out
    episode_reward = 0

    while not (terminated or truncated):

        action = select_action(state_bucket, episode)

        obs, reward, terminated, truncated, _ = env.step(action)

        new_state_bucket = get_state_bucket(obs, config.OBS_LOWS, config.BUCKET_SIZES)

        if terminated:
            reward = config.GOAL_REWARD
            print(f"Done episode {episode} with reward {episode_reward + reward}")
        # elif truncated:
        #     reward = -100.0

        episode_reward += reward

        if terminated:
            # by definition, taking terminal state q as zero
            max_future_q = 0.0
        else:
            max_future_q = np.max(q_table[new_state_bucket])

        current_q = q_table[state_bucket + (action,)]

        new_q = (1.0 - config.LEARNING_RATE) * current_q + config.LEARNING_RATE * (reward + config.DISCOUNT * max_future_q)

        q_table[state_bucket + (action,)] = new_q

        state_bucket = new_state_bucket

    rewards[episode] = episode_reward

    # update exploration rate
    if config.EXPLORATION > 0.0:
        if config.START_EXPLORING <= episode <= config.END_EXPLORING:
            explorations[episode] = config.EXPLORATION
            config.EXPLORATION = max(config.EPS_LOW, config.EXPLORATION - config.EXPLORATION_DECAY)

    # consider exploring, if a solution is continuously exploited.
    if np.min(rewards[max(0, episode - (config.QTABLE_SAVE_LOOKBACK - 1)):episode + 1]) >= 0:
        # np.save(f"q_tables/q_table6_{episode}.npy", q_table)

        if config.EXPLORATION > 0.0:
            if config.EPISODES // 5 <= episode <= config.START_EXPLORING:
                np.save(f"{QTABLES_DIR}/exploited.npy", q_table)
                print("started exploring!")
                config.START_EXPLORING = episode
                config.EXPLORATION_DECAY = config.EXPLORATION / (config.END_EXPLORING - config.START_EXPLORING)

    # save q table periodically
    if (episode % config.QTABLE_SAVE_PERIOD) == 0:
        np.save(f"{QTABLES_DIR}/{episode}.npy", q_table)

env.close()

np.save(f"{QTABLES_DIR}/final.npy", q_table)
np.save(f"{RESULTS_DIR}/rewards.npy", rewards)
np.save(f"{RESULTS_DIR}/explorations.npy", explorations)
print("q table and results saved!")
