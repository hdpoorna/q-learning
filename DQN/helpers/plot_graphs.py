"""
py311
hdpoorna
"""

# import packages
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

MODEL_ID = "adam-huber-default-2023-07-12-17-46-05"

# REWARDS
rewards = pd.Series(np.load(f"../results/{MODEL_ID}/rewards.npy"))
x = list(range(0, len(rewards), 1))

window_size = len(rewards)//100
rewards = pd.concat([pd.Series(np.zeros(window_size-1)-rewards.min()), rewards])
windows = rewards.rolling(window=window_size, step=1)

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
plt.savefig(f"../results/{MODEL_ID}/rewards.svg")
# plt.close()
print("rewards.svg saved")

# EXPLORATION
explorations = np.load(f"../results/{MODEL_ID}/explorations.npy")

plt.figure()
plt.plot(x, explorations, color="b")
plt.title('Exploration')
plt.xlabel("Episode")
plt.ylabel("Rate")
plt.ylim(None, 1)
plt.xticks(list(range(0, len(x)+1, window_size*10)))
plt.savefig(f"../results/{MODEL_ID}/explorations.svg")
# plt.close()
print("explorations.svg saved")

# LOSS

losses = np.load(f"../results/{MODEL_ID}/losses.npy")

loss_mins = np.zeros(len(losses))
loss_maxs = np.zeros(len(losses))
loss_means = np.zeros(len(losses))

for i in range(len(losses)):
    row = losses[i]
    non_zero = row[row != 0.0]
    loss_mins[i] = np.min(row)
    loss_maxs[i] = np.max(row)
    loss_means[i] = np.mean(row)

START = 11
END = len(losses)

xs = list(range(START, END))

plt.figure()
plt.plot(xs, loss_mins[START:END], label="min", color="g")
plt.plot(xs, loss_maxs[START:END], label="max", color="r")
plt.plot(xs, loss_means[START:END], label="mean", color="b")
plt.legend(loc="upper left")
plt.title('Loss')
plt.xlabel("Episode")
plt.ylabel("Loss")
# plt.xticks(list(range(0, len(x)+1, window_size*10)))
plt.savefig(f"../results/{MODEL_ID}/loss.svg")
# plt.close()
print("loss.svg saved")

plt.show()
