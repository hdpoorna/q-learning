"""
py311
hdpoorna
"""

# import packages
import gymnasium as gym
import numpy as np
from helpers import config
from helpers.q_table_helper import *

# make the env
env = gym.make("MountainCar-v0", render_mode="human")

# set constants
config.OBS_HIGHS = env.observation_space.high
config.OBS_LOWS = env.observation_space.low

# load q table
QTABLE_ID = "default-2023-07-13-04-55-10"
q_table = load_q_table(f"q_tables/{QTABLE_ID}/final.npy")
config.NUM_BUCKETS = list(q_table.shape[:-1])
config.BUCKET_SIZES = (config.OBS_HIGHS - config.OBS_LOWS)/config.NUM_BUCKETS

# initial state
state_bucket = get_state_bucket(env.reset()[0], config.OBS_LOWS, config.BUCKET_SIZES)

terminated = False      # goal achieved
truncated = False       # timed out

while not (terminated or truncated):

    action = np.argmax(q_table[state_bucket])
    obs, reward, terminated, truncated, _ = env.step(action)
    env.render()
    new_state_bucket = get_state_bucket(obs, config.OBS_LOWS, config.BUCKET_SIZES)
    state_bucket = new_state_bucket

env.close()
