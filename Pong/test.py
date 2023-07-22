"""
py311
hdpoorna
"""

# import packages
import os
import imageio
import cv2
from datetime import datetime
from pongWrapper import *
import numpy as np
import tensorflow as tf
from helpers import config
from helpers.dqn_helper import *

# make the env
env = PongWrapper(render_mode="rgb_array", points_per_episode=config.POINTS_PER_EPISODE)

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

frames_orig = [env.render()]
frames_mod = [env.render_cv(current_state)]

while not (terminated or truncated):
    qs = policy_model(inputs=[scale_states(current_state)], training=False)
    # print(qs)
    action = tf.argmax(qs[0])
    # print(action)
    obs, reward, terminated, truncated, _ = env.step(action.numpy())

    frames_orig.append(env.render())
    frame_rgb = env.render_cv(obs)
    frames_mod.append(frame_rgb)

    current_state = obs

results_dir = f"results/{MODEL_ID}"
make_dir(results_dir)

now_utc = datetime.utcnow()
now_str = now_utc.strftime("%Y-%m-%d-%H-%M-%S")
orig_path = "{}/{}-orig.gif".format(results_dir, now_str)
mod_path = "{}/{}-mod.gif".format(results_dir, now_str)

imageio.mimsave(orig_path, frames_orig, duration=1000/env.MAX_FPS, loop=0)
imageio.mimsave(mod_path, frames_mod, duration=1000/env.MAX_FPS, loop=0)
print(f"gifs saved to {results_dir}")

cv2.destroyAllWindows()
env.close()

