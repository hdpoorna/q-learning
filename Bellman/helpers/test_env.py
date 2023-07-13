"""
py311
hdpoorna
"""

import gymnasium as gym

env = gym.make("MountainCar-v0")
# env = gym.make("MountainCar-v0", render_mode="human")
env.reset()

print(env.action_space.n)
print(env.observation_space.high)
print(env.observation_space.low)
print(env.goal_position)

terminated = False
truncated = False
while not (terminated or truncated):
    action = 2
    obs, reward, terminated, truncated, _ = env.step(action)
    break
    # env.render()

env.close()
