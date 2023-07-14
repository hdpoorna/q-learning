"""
py311
hdpoorna
"""

import os
import numpy as np


def init_q_table(env, init_dist="uniform"):
    obs_space_sizes = env.observation_space.high - env.observation_space.low
    bucket_sizes = np.power(10.0, np.floor(np.log10(obs_space_sizes)) - 1.0)
    num_buckets = list((obs_space_sizes / bucket_sizes).astype(int))
    if init_dist == "normal":
        q_table = np.random.normal(loc=-1.0, scale=0.5, size=num_buckets + [env.action_space.n])
    else:
        q_table = np.random.uniform(low=-2.0, high=0.0, size=num_buckets + [env.action_space.n])
    print("q table initialized!")
    return q_table


def load_q_table(path):
    if os.path.exists(path):
        q_table = np.load(path)
        print("q table loaded!")
        return q_table
    else:
        exit(f"{path} Not Found!")


def get_state_bucket(state, obs_lows, bucket_sizes):
    bucket = (state - obs_lows)/bucket_sizes
    return tuple(bucket.astype(int))


def make_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        print(f"created directory {dir_path}")


def config_to_lines():

    config_path = os.path.join("helpers", "config.py")

    with open(config_path, "r") as f:
        config_lines = f.readlines()

    return config_lines[5:]


def write_to_txt(q_tabel_id):

    config_lines = config_to_lines()

    config_max_len = len(max(config_lines, key=len))

    results_dir = os.path.join("results", "{}".format(q_tabel_id))
    make_dir(results_dir)

    txt_path = os.path.join(results_dir, "{}.txt".format(q_tabel_id))

    with open(txt_path, "w") as f:
        # write model_id
        f.write("\nMODEL_ID: {}\n".format(q_tabel_id))
        f.write("\n{}\n".format("-" * config_max_len))

        # write config
        f.write("\nCONFIG\n")
        f.write("".join(config_lines))
        f.write("\n{}\n".format("-" * config_max_len))

    print("Text file saved to {}".format(txt_path))
