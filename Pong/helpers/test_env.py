"""
py311
hdpoorna
"""

import numpy as np
import gymnasium as gym


env = gym.make("ALE/Pong-v5", mode=0, difficulty=0, obs_type="rgb", full_action_space=False, render_mode="human")
env.metadata["render_fps"] = 24
init_state, _ = env.reset()

NUM_ACTIONS = env.action_space.n
print(NUM_ACTIONS)

terminated = False
truncated = False
steps = 0
reward = 0.0
while not (terminated or truncated):
    action = np.random.randint(low=0, high=NUM_ACTIONS, size=1, dtype=int)
    obs, reward, terminated, truncated, _ = env.step(action[0])
    steps += 1
    env.render()
    if reward != 0.0:
        print(steps)
        # env.reset()
        steps = 0
        # break

env.close()
