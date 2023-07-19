"""
py311
hdpoorna
"""

import gymnasium as gym
from gymnasium.utils.play import play

env = gym.make("MountainCar-v0", render_mode="rgb_array")
env.metadata["render_fps"] = 24

NUM_ACTIONS = env.action_space.n
print(NUM_ACTIONS)

# key_action_map = {str(k): k for k in range(NUM_ACTIONS)}

key_action_map = {
    "a": 0,
    "s": 1,
    "d": 2
}

play(env, keys_to_action=key_action_map, noop=1)

