"""
py311
hdpoorna
"""

# import packages
import numpy as np
import cv2
import gymnasium as gym


class PongWrapper:

    _DEFAULT_REWARD = 0.0
    _LOWEST_SIDE = 40
    _ACTION_SPACE_SIZE = 3
    _AVG_STEP_LIMIT_PER_POINT = 500
    _ACTION_MAP = {
        0: 0,   # NOOP
        1: 2,   # UP
        2: 3    # DOWN
    }
    _PONG_COLORS = {
        "BG": [144, 72, 17],
        "BALL": [236, 236, 236],
        "OPP": [213, 130, 74],
        "AGENT": [92, 186, 92]
    }
    _NEW_COLORS = {
        "BG": [0, 0, 0],
        "BALL": [255, 255, 255],
        "OPP": [255, 0, 0],
        "AGENT": [0, 0, 255]
    }
    _PLAYING_AREA = {
        "BEGIN": 34,
        "END": 194
    }
    MAX_FPS = 100

    def __init__(self, render_mode=None, points_per_episode=3):

        assert points_per_episode % 2 == 1, "Need odd points per episode to find winner."

        self._points_per_episode = points_per_episode
        self._step_limit_per_episode = self._points_per_episode * self._AVG_STEP_LIMIT_PER_POINT

        self._render_mode = render_mode

        self._env = gym.make("ALE/Pong-v5", mode=0, difficulty=0, obs_type="rgb", full_action_space=False, render_mode=self._render_mode)
        if self._render_mode == "human":
            self._env.metadata["render_fps"] = 30

        self._step_count = None

        self._terminated = None
        self._truncated = None

        self._episode_tot_abs_reward = None

    def reset(self):

        self._step_count = 0
        self._terminated = False
        self._truncated = False
        self._episode_tot_abs_reward = 0

        obs, info = self._env.reset()

        return self._preprocess_obs(obs), info

    def close(self):

        assert self._step_count is not None, "reset method should be called first to initialize the environment"

        self._step_count = None

        self._terminated = None
        self._truncated = None

        self._episode_tot_abs_reward = None

        self._env.close()

    def step(self, action):

        assert self._step_count is not None, "reset method should be called first to initialize the environment"
        assert action in list(range(self._ACTION_SPACE_SIZE)), f"action should be one of {list(range(self._ACTION_SPACE_SIZE))}"
        assert not self._terminated, "The episode was terminated. Call reset method to re-initialize."
        assert not self._truncated, "The episode timed out. Call reset method to re-initialize."

        self._step_count += 1

        obs, reward, self._terminated, self._truncated, info = self._env.step(self._ACTION_MAP[action])

        reward = float(reward)
        if self._DEFAULT_REWARD == 1.0:
            new_reward = 1.0
            if reward > 0.0:
                self._episode_tot_abs_reward += 1
                new_reward += self._AVG_STEP_LIMIT_PER_POINT
                # self._terminated = True
            elif reward < 0.0:
                self._episode_tot_abs_reward += 1
                new_reward -= self._AVG_STEP_LIMIT_PER_POINT
            reward = new_reward
        else:
            if reward != 0.0:
                self._episode_tot_abs_reward += 1

        if self._episode_tot_abs_reward >= self._points_per_episode:
            self._truncated = True

        if self._step_count >= self._step_limit_per_episode:
            self._truncated = True

        return self._preprocess_obs(obs), reward, self._terminated, self._truncated, info

    def _preprocess_obs(self, obs):
        out = obs[self._PLAYING_AREA["BEGIN"]:self._PLAYING_AREA["END"]]                    # crop scores -> (160, 160)
        """
        ball, opp and agent had sides with multiples of 4. 
        can decimate the image by 4 without losing information.
        """
        out = cv2.resize(out, (self._LOWEST_SIDE, self._LOWEST_SIDE), interpolation=cv2.INTER_NEAREST_EXACT)
        out[(out == self._PONG_COLORS["BG"]).all(axis=2)] = self._NEW_COLORS["BG"]          # bg to black
        out[(out == self._PONG_COLORS["BALL"]).all(axis=2)] = self._NEW_COLORS["BALL"]      # ball to white
        out[(out == self._PONG_COLORS["OPP"]).all(axis=2)] = self._NEW_COLORS["OPP"]        # opp to red
        out[(out == self._PONG_COLORS["AGENT"]).all(axis=2)] = self._NEW_COLORS["AGENT"]    # agent to blue
        return out

    def render(self):
        assert self._step_count is not None, "reset method should be called first to initialize the environment"
        assert self._render_mode == "human", "render_mode is not set to human. Recreate."

        self._env.render()

    def render_cv(self, obs):
        assert self._step_count is not None, "reset method should be called first to initialize the environment"

        img_rgb = cv2.resize(obs, (320, 320), interpolation=cv2.INTER_NEAREST)
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        cv2.imshow("Pong", img_bgr)
        cv2.waitKey(1000 // self.MAX_FPS)


if __name__ == "__main__":
    env = PongWrapper(render_mode=None, points_per_episode=1)
    env.MAX_FPS = 100

    init_state, _ = env.reset()

    terminated = False
    truncated = False
    steps = 0
    reward = 0.0
    while not (terminated or truncated):
        action = np.random.randint(low=0, high=env._ACTION_SPACE_SIZE, size=1, dtype=int)
        obs, reward, terminated, truncated, _ = env.step(action[0])
        ball = np.count_nonzero(obs[:, :, 1])
        opp = np.count_nonzero(obs[:, :, 0]) - ball
        agent = np.count_nonzero(obs[:, :, 2]) - ball
        print(f"agent: {agent}, opp:{opp}, ball:{ball}")
        # print(reward)
        steps += 1
        env.render_cv(obs)
        if reward != env._DEFAULT_REWARD:
            print(steps)
            # env.reset()
            steps = 0
            # break

    cv2.destroyAllWindows()
    env.close()

