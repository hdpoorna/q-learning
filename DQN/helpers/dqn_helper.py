"""
py311
hdpoorna
"""

import os
import math
import random
from collections import namedtuple, deque
from itertools import islice
import tensorflow as tf
import numpy as np

# to record 1 transition
Transition = namedtuple('Transition',
                        ('current_state', 'action', 'next_state', 'reward', 'terminated'))


# to collect most recent past transitions into a memory buffer
class ReplayBuffer:

    def __init__(self, buffer_size):
        self.buffer = deque([], maxlen=buffer_size)

    def push(self, current_state, action, next_state, reward, terminated=False):
        self.buffer.append(Transition(current_state, action, next_state, reward, terminated))

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def sample_from_end(self, batch_size):
        return list(islice(self.buffer, self.__len__()-batch_size, self.__len__()))

    def __len__(self):
        return len(self.buffer)


def DenseModel(n_obs=2, n_actions=3, dropout_rate=0.0, deeper=False, init_weights_with="glorot_uniform"):

    if init_weights_with == "random_normal":
        initializer = tf.keras.initializers.RandomNormal(mean=-1.0, stddev=0.5)
    elif init_weights_with == "random_uniform":
        initializer = tf.keras.initializers.RandomUniform(minval=-2.0, maxval=0.0)
    else:
        initializer = tf.keras.initializers.GlorotUniform(seed=42)

    neurons = [128, 128, 128]

    model = tf.keras.Sequential(name="DenseModel")
    model.add(tf.keras.Input(shape=(n_obs,), name="state"))

    if deeper:
        model.add(tf.keras.layers.Dense(neurons[0], activation="relu", kernel_initializer=initializer))
        if dropout_rate > 0.0:
            model.add(tf.keras.layers.Dropout(dropout_rate))

    model.add(tf.keras.layers.Dense(neurons[1], activation="relu", kernel_initializer=initializer))
    if dropout_rate > 0.0:
        model.add(tf.keras.layers.Dropout(dropout_rate))

    model.add(tf.keras.layers.Dense(neurons[2], activation="relu", kernel_initializer=initializer))
    if dropout_rate > 0.0:
        model.add(tf.keras.layers.Dropout(dropout_rate))

    model.add(tf.keras.layers.Dense(n_actions, name="action_q", kernel_initializer=initializer))

    print(model.summary())

    return model


def bell_curve_value(x, mean=0.0, sd=1.0):
    # an exploration strategy
    return 1.0/(sd * np.sqrt(2 * np.pi)) * np.exp(-(x - mean)**2 / (2 * sd**2))


def get_exp_decayed(episode, total_episodes):
    # an exploration strategy
    x = 5.9 * episode / total_episodes
    return 1.0 - 1.0/(1.0 + np.exp(2.95 - x))


def get_exp_cos(episode, total_episodes, start_high=True):
    # an exploration strategy
    if start_high:
        x = 3.0 * np.pi * episode / total_episodes
        return 0.5 + 0.45 * np.cos(x)
    else:
        x = 4.0 * np.pi * episode / total_episodes
        return 0.5 - 0.45 * np.cos(x)


def scale_states(state, lows, highs):
    return (state - lows)/(highs - lows)


def make_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        print(f"created directory {dir_path}")


def model_summary_to_lines(model, line_length=None):
    line_lst = []
    model.summary(print_fn=lambda x: line_lst.append(x), line_length=line_length)
    return line_lst


def config_to_lines():

    config_path = os.path.join("helpers", "config.py")

    with open(config_path, "r") as f:
        config_lines = f.readlines()

    return config_lines[5:]


def write_to_txt(model_id, model_summary):

    summary_max_len = len(max(model_summary, key=len))

    results_dir = os.path.join("results", "{}".format(model_id))
    make_dir(results_dir)

    txt_path = os.path.join(results_dir, "{}.txt".format(model_id))

    with open(txt_path, "w") as f:
        # write model_id
        f.write("\nMODEL_ID: {}\n".format(model_id))
        f.write("\n{}\n".format("-" * summary_max_len))

        # write model_summary
        f.write("\nMODEL_SUMMARY\n")
        f.write("\n".join(model_summary))

        # write config
        f.write("\nCONFIG\n")
        f.write("".join(config_to_lines()))
        f.write("\n{}\n".format("-" * summary_max_len))

    print("Text file saved to {}".format(txt_path))


if __name__ == "__main__":
    buffer = ReplayBuffer(buffer_size=32)
