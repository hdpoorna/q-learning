"""
py311
hdpoorna
"""

# import packages
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

QTABLE_ID = "default-2023-07-13-08-51-31"

# REWARDS
rewards = pd.Series(np.load(f"../results/{QTABLE_ID}/rewards.npy"))
x = list(range(0, len(rewards), 1))

window_size = len(rewards)//100

rewards = pd.concat([pd.Series(np.zeros(window_size-1)-rewards.min()), rewards])
windows = rewards.rolling(window=window_size, step=1)
# windows = rewards.rolling(window=window_size, step=window_size)

# moving stats
reward_means = windows.mean().dropna().values
reward_mins = windows.min().dropna().values
reward_maxs = windows.max().dropna().values

plt.figure()
# plt.plot(rewards, label="mean", color="k")
plt.plot(x, reward_means, label="mean", color="b")
plt.plot(x, reward_mins, label="min", color="r")
plt.plot(x, reward_maxs, label="max", color="g")
plt.legend(loc="upper left")
plt.title('Rewards')
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.xticks(list(range(0, len(rewards)+1, window_size*10)))
plt.savefig(f"../results/{QTABLE_ID}/rewards.svg")
# plt.close()
print("rewards.svg saved")

# EXPLORATION
explorations = np.load(f"../results/{QTABLE_ID}/explorations.npy")

plt.figure()
plt.plot(x, explorations, color="b")
plt.title('Exploration')
plt.xlabel("Episode")
plt.ylabel("Rate")
plt.ylim(None, 1)
plt.xticks(list(range(0, len(x)+1, window_size*10)))
plt.savefig(f"../results/{QTABLE_ID}/explorations.svg")
# plt.close()
print("explorations.svg saved")

plt.show()
