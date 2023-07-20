"""
py311
hdpoorna
"""

import gymnasium as gym
from gymnasium.utils.play import play

env = gym.make("ALE/Pong-v5", mode=0, difficulty=0, obs_type="rgb", full_action_space=False, render_mode="rgb_array")
env.metadata["render_fps"] = 24

NUM_ACTIONS = env.action_space.n
print(NUM_ACTIONS)

# key_action_map = {str(k): k for k in range(NUM_ACTIONS)}

key_action_map = {
    "a": 0,
    "1": 1,
    "w": 2,
    "s": 3,
    "4": 4,
    "5": 5
}

play(env, keys_to_action=key_action_map, noop=0)

