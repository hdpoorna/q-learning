"""
py311
hdpoorna
"""

# import packages
import os
import imageio
import cv2
from datetime import datetime
import gymnasium as gym
import numpy as np
import tensorflow as tf
from helpers import config
from helpers.dqn_helper import *

# make the env
env = gym.make("MountainCar-v0", render_mode="rgb_array")
env.metadata["render_fps"] = 30

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

frames = [env.render()]

while not (terminated or truncated):
    qs = policy_model([scale_states(current_state, lows=config.OBS_LOWS, highs=config.OBS_HIGHS)])
    # print(qs)
    action = tf.argmax(qs[0])
    # print(action)
    obs, reward, terminated, truncated, _ = env.step(action.numpy())

    frame_rgb = env.render()
    frames.append(frame_rgb)
    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    cv2.imshow("Car", frame_bgr)
    cv2.waitKey(1000 // env.metadata["render_fps"])

    current_state = obs

results_dir = f"results/{MODEL_ID}"
make_dir(results_dir)

now_utc = datetime.utcnow()
now_str = now_utc.strftime("%Y-%m-%d-%H-%M-%S")
gif_path = "{}/{}.gif".format(results_dir, now_str)

imageio.mimsave(gif_path, frames, duration=1000/env.metadata["render_fps"], loop=0)
print(f"gif saved to {gif_path}")

cv2.destroyAllWindows()
env.close()

