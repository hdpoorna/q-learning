"""
py311
hdpoorna
"""

# import packages
import os
from pongWrapper import *
import numpy as np
import tensorflow as tf
from helpers import config
from helpers.dqn_helper import *

# make the env
env = PongWrapper(render_mode="human", points_per_episode=config.POINTS_PER_EPISODE)

# set constants
config.NUM_ACTIONS = env._ACTION_SPACE_SIZE

# load the model
MODEL_ID = "pong-conv-greedy-2023-07-20-16-53-20"
MODEL_EPISODE = "73"
model_dir_path = os.path.join("saved_models", f"{MODEL_ID}/model-{MODEL_EPISODE}")
policy_model = tf.saved_model.load(model_dir_path)

# initial state
current_state = env.reset()[0]

terminated = False      # goal achieved
truncated = False       # timed out

while not (terminated or truncated):
    qs = policy_model(inputs=[scale_states(current_state)], training=False)
    # print(qs)
    action = tf.argmax(qs[0])
    # print(action)
    obs, reward, terminated, truncated, _ = env.step(action.numpy())
    env.render()
    current_state = obs

env.close()
