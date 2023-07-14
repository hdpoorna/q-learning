"""
py311
hdpoorna
"""

# import packages
import os
import gymnasium as gym
import numpy as np
import tensorflow as tf
from helpers import config
from helpers.dqn_helper import *

# make the env
env = gym.make("MountainCar-v0", render_mode="human")

# set constants
config.OBS_HIGHS = env.observation_space.high
config.OBS_LOWS = env.observation_space.low

# load the model
MODEL_ID = "adam-huber-default-2023-07-12-17-46-05"
MODEL_EPISODE = "final"
model_dir_path = os.path.join("saved_models", f"{MODEL_ID}/model-{MODEL_EPISODE}")
policy_model = tf.saved_model.load(model_dir_path)

# initial state
current_state = env.reset()[0]

terminated = False      # goal achieved
truncated = False       # timed out

while not (terminated or truncated):
    qs = policy_model([scale_states(current_state, lows=config.OBS_LOWS, highs=config.OBS_HIGHS)])
    # print(qs)
    action = tf.argmax(qs[0])
    # print(action)
    obs, reward, terminated, truncated, _ = env.step(action.numpy())
    env.render()
    current_state = obs

env.close()

