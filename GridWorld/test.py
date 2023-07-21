"""
py311
hdpoorna
"""

# import packages
import os
import imageio
import cv2
from datetime import datetime
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
env.MAX_FPS = 6

# load the model
MODEL_ID = "gw-conv-through-2023-07-18-17-32-23"
MODEL_EPISODE = "final"
model_dir_path = os.path.join("saved_models", f"{MODEL_ID}/model-{MODEL_EPISODE}")
policy_model = tf.saved_model.load(model_dir_path)

# initial state
current_state = env.reset()[0]

terminated = False      # goal achieved
truncated = False       # timed out

frames = [env.render()]

while not (terminated or truncated):
    qs = policy_model(inputs=[scale_states(current_state)], training=False)
    # print(qs)
    action = tf.argmax(qs[0])
    # print(action)
    obs, reward, terminated, truncated, _ = env.step(action.numpy())

    frame_rgb = env.render()
    frames.append(frame_rgb)

    current_state = obs

results_dir = f"results/{MODEL_ID}"
make_dir(results_dir)

now_utc = datetime.utcnow()
now_str = now_utc.strftime("%Y-%m-%d-%H-%M-%S")
gif_path = "{}/{}.gif".format(results_dir, now_str)

imageio.mimsave(gif_path, frames, duration=1000/env.MAX_FPS, loop=0)
print(f"gif saved to {gif_path}")

cv2.destroyAllWindows()
env.close()

