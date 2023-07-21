"""
py311
hdpoorna
"""

# import packages
import imageio
import cv2
from datetime import datetime
import gymnasium as gym
import numpy as np
from helpers import config
from helpers.q_table_helper import *

# make the env
env = gym.make("MountainCar-v0", render_mode="rgb_array")
env.metadata["render_fps"] = 30

# set constants
config.OBS_HIGHS = env.observation_space.high
config.OBS_LOWS = env.observation_space.low

# load q table
QTABLE_ID = "default-2023-07-13-08-51-31"
q_table = load_q_table(f"q_tables/{QTABLE_ID}/final.npy")
config.NUM_BUCKETS = list(q_table.shape[:-1])
config.BUCKET_SIZES = (config.OBS_HIGHS - config.OBS_LOWS)/config.NUM_BUCKETS

# initial state
state_bucket = get_state_bucket(env.reset()[0], config.OBS_LOWS, config.BUCKET_SIZES)

terminated = False      # goal achieved
truncated = False       # timed out

frames = [env.render()]

while not (terminated or truncated):

    action = np.argmax(q_table[state_bucket])
    obs, reward, terminated, truncated, _ = env.step(action)

    frame_rgb = env.render()
    frames.append(frame_rgb)
    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    cv2.imshow("Car", frame_bgr)
    cv2.waitKey(1000 // env.metadata["render_fps"])

    new_state_bucket = get_state_bucket(obs, config.OBS_LOWS, config.BUCKET_SIZES)
    state_bucket = new_state_bucket


results_dir = f"results/{QTABLE_ID}"
make_dir(results_dir)

now_utc = datetime.utcnow()
now_str = now_utc.strftime("%Y-%m-%d-%H-%M-%S")
gif_path = "{}/{}.gif".format(results_dir, now_str)

imageio.mimsave(gif_path, frames, duration=1000/env.metadata["render_fps"], loop=0)
print(f"gif saved to {gif_path}")

cv2.destroyAllWindows()
env.close()
