"""
py311
hdpoorna
"""

# import packages
import os
import random
from datetime import datetime
from tqdm import tqdm
import gymnasium as gym
import numpy as np
import tensorflow as tf
from helpers import config
from helpers.dqn_helper import *

# make the env
env = gym.make("MountainCar-v0")

# set constants
config.OBS_HIGHS = env.observation_space.high
config.OBS_LOWS = env.observation_space.low
config.NUM_ACTIONS = env.action_space.n
config.ALL_ACTIONS = np.array(list(range(config.NUM_ACTIONS)))
config.GOAL_POSITION = env.goal_position
config.NUM_OBS = len(config.OBS_HIGHS)

# initialize models
policy_model = DenseModel(n_obs=config.NUM_OBS, n_actions=config.NUM_ACTIONS)
target_model = DenseModel(n_obs=config.NUM_OBS, n_actions=config.NUM_ACTIONS)

policy_model.build(input_shape=(None, config.NUM_OBS))
target_model.build(input_shape=(None, config.NUM_OBS))

target_model.set_weights(policy_model.get_weights())

# policy_optimizer = tf.keras.optimizers.AdamW(learning_rate=config.LEARNING_RATE, amsgrad=False)
policy_optimizer = tf.keras.optimizers.Adam(learning_rate=config.LEARNING_RATE)
policy_loss_func = tf.keras.losses.Huber(reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE)
# policy_loss_func = tf.keras.losses.MeanSquaredError()

# policy_model.compile(optimizer=policy_optimizer, loss=policy_loss_func)

# initialize the memory buffer
replay_buffer = ReplayBuffer(buffer_size=config.BATCH_SIZE ** 2)


now_utc = datetime.utcnow()
now_str = now_utc.strftime("%Y-%m-%d-%H-%M-%S")
config.MODEL_ID = "{}-{}".format(config.MODEL_ID, now_str)
MODELS_DIR = os.path.join("saved_models", config.MODEL_ID)
RESULTS_DIR = os.path.join("results", config.MODEL_ID)
make_dir(MODELS_DIR)
make_dir(RESULTS_DIR)


# write details to txt
model_summary = model_summary_to_lines(policy_model)
write_to_txt(config.MODEL_ID, model_summary)


@tf.function
def train_step(batch_current_state, batch_action, batch_next_state, batch_reward, batch_terminated):
    future_qs = target_model(inputs=[batch_next_state], training=False)
    max_future_qs = tf.reduce_max(future_qs, axis=-1, keepdims=True)

    terminal_qs = tf.zeros_like(max_future_qs)      # by definition, taking terminal state q as zero
    max_future_qs = tf.where(batch_terminated, terminal_qs, max_future_qs)
    new_qs = batch_reward + config.DISCOUNT * max_future_qs

    # rows = tf.convert_to_tensor(list(range(len(batch_action))), dtype=tf.int64)
    rows = tf.range(config.BATCH_SIZE, dtype=tf.int64)
    action_indices = tf.stack([rows, batch_action], axis=-1)

    with tf.GradientTape() as policyTape:
        policyTape.watch(policy_model.trainable_variables)
        current_qs = policy_model(inputs=[batch_current_state], training=True)
        action_qs = tf.expand_dims(tf.gather_nd(current_qs, action_indices), axis=-1)
        loss = policy_loss_func(new_qs, action_qs)
        # tf.print("new", loss)

    policy_gradients = policyTape.gradient(loss, policy_model.trainable_variables)
    # clipped_gradients = [(tf.clip_by_value(gradient, -100.0, 100.0)) for gradient in policy_gradients]
    policy_optimizer.apply_gradients(zip(policy_gradients, policy_model.trainable_variables))

    return loss


def update_target_model(episode, soft=True):
    if soft:
        policy_weights = policy_model.get_weights()
        target_weights = target_model.get_weights()

        target_model.set_weights([(1.0 - config.TARGET_NET_LR) * target_weights[i] +
                                  config.TARGET_NET_LR * policy_weights[i] for i in range(len(policy_weights))])

        # print("new", policy_weights[5])
    elif episode % 5 == 0:
        target_model.set_weights(policy_model.get_weights())
        print("target model updated!")


def select_action(state, episode):
    if config.EXPLORATION > 0.0:
        if config.START_EXPLORING <= episode <= config.END_EXPLORING:
            if np.random.random() <= config.EXPLORATION:
                action = tf.convert_to_tensor(np.random.choice(config.ALL_ACTIONS, 1)[0], dtype=tf.int64)
            else:
                action = tf.argmax(policy_model(inputs=[tf.expand_dims(state, axis=0)], training=False)[0])
        else:
            action = tf.argmax(policy_model(inputs=[tf.expand_dims(state, axis=0)], training=False)[0])

    else:
        action = tf.argmax(policy_model(inputs=[tf.expand_dims(state, axis=0)], training=False)[0])
    return tf.expand_dims(action, axis=0)


rewards = np.zeros(config.EPISODES, dtype=np.float32)
losses = np.zeros([config.EPISODES, config.MAX_TIME_STEPS], dtype=np.float32)
explorations = np.zeros(config.EPISODES, dtype=np.float32)


print("training starting!")
for episode in tqdm(range(config.EPISODES), ascii=True, unit="episodes"):
    # get initial state
    current_state = tf.convert_to_tensor(scale_states(env.reset()[0], lows=config.OBS_LOWS, highs=config.OBS_HIGHS),
                                         dtype=tf.float32)

    terminated = False      # goal achieved
    truncated = False       # timed out
    episode_reward = 0

    step = 0

    while not (terminated or truncated):

        action = select_action(current_state, episode)
        # greedy
        # action = tf.expand_dims(tf.argmax(policy_model(inputs=[tf.expand_dims(current_state, axis=0)], training=False)[0]), axis=0)

        obs, reward, terminated, truncated, _ = env.step(action.numpy()[0])

        if terminated:
            reward = config.GOAL_REWARD
            print(f"Done episode {episode} with reward {episode_reward + reward}")
        # elif truncated:
        #     reward = -100.0

        episode_reward += reward

        next_state = tf.convert_to_tensor(scale_states(obs, lows=config.OBS_LOWS, highs=config.OBS_HIGHS),
                                          dtype=tf.float32)

        replay_buffer.push(current_state=current_state,
                           action=action,
                           next_state=next_state,
                           reward=tf.convert_to_tensor([float(reward)], dtype=tf.float32),
                           terminated=tf.convert_to_tensor([terminated])
                           )

        current_state = next_state

        if len(replay_buffer) < (config.BATCH_SIZE ** 2) / 2:
            continue

        if terminated:
            # making sure terminated transition is included in the batch
            transitions = replay_buffer.sample_from_end(config.BATCH_SIZE)
        else:
            transitions = replay_buffer.sample(config.BATCH_SIZE)
        batch = Transition(*zip(*transitions))

        batch_current_state = tf.stack(batch.current_state, axis=0)
        batch_action = tf.concat(batch.action, axis=0)
        batch_next_state = tf.stack(batch.next_state, axis=0)
        batch_reward = tf.stack(batch.reward, axis=0)
        batch_terminated = tf.stack(batch.terminated, axis=0)

        if terminated:
            # forcing the model to see the terminated transition multiple times
            for _ in range(config.BATCH_SIZE):
                step_loss = train_step(batch_current_state, batch_action, batch_next_state, batch_reward, batch_terminated)
                update_target_model(episode=episode, soft=True)
        else:
            step_loss = train_step(batch_current_state, batch_action, batch_next_state, batch_reward, batch_terminated)
            update_target_model(episode=episode, soft=True)

        losses[episode, step] = step_loss.numpy()
        step += 1

    rewards[episode] = episode_reward

    # update exploration rate
    if config.EXPLORATION > 0.0:
        if config.START_EXPLORING <= episode <= config.END_EXPLORING:
            explorations[episode] = config.EXPLORATION
            config.EXPLORATION = max(config.EPS_END, config.EXPLORATION - config.EXPLORATION_DECAY)

    # overwrite results every episode, in case training is interrupted
    np.save(f"{RESULTS_DIR}/rewards.npy", rewards)
    np.save(f"{RESULTS_DIR}/losses.npy", losses)
    np.save(f"{RESULTS_DIR}/explorations.npy", explorations)

    if not (len(replay_buffer) < (config.BATCH_SIZE ** 2) / 2):
        if np.min(rewards[max(0, episode - (config.MODEL_SAVE_LOOKBACK - 1)):episode + 1]) >= 0:
            # save models that are continuously successful
            model_dir_path = os.path.join(MODELS_DIR, f"model-{episode}")
            tf.saved_model.save(policy_model, model_dir_path)
            print(f"model saved to {model_dir_path}")

            # consider exploring, if a solution is continuously exploited.
            if config.EXPLORATION > 0.0:
                if config.EPISODES // 5 <= episode <= config.START_EXPLORING:
                    print("started exploring!")
                    config.START_EXPLORING = episode
                    config.EXPLORATION_DECAY = config.EXPLORATION / (config.END_EXPLORING - config.START_EXPLORING)

env.close()

model_dir_path = os.path.join(MODELS_DIR, f"model-final")
tf.saved_model.save(policy_model, model_dir_path)
print("final model saved!")
