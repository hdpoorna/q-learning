"""
py311
hdpoorna
"""

# import packages
import os
import cv2
from gridWorld import *
import numpy as np
import tensorflow as tf
from helpers import config
from helpers.dqn_helper import *

# make the env
env = GridWorld(wall_rule=WallRule.THROUGH)

# set constants
config.NUM_ACTIONS = env._ACTION_SPACE_SIZE
config.MAX_TIME_STEPS = env.step_limit
env.MAX_FPS = 24

# load the model
MODEL_ID = "gw-conv-through-2023-07-18-17-32-23"
MODEL_EPISODE = "final"
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
    print(action)
    obs, reward, terminated, truncated, _ = env.step(action.numpy())
    env.render()
    current_state = obs

cv2.destroyAllWindows()
env.close()

