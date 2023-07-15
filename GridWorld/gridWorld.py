"""
py311
hdpoorna
"""

# import packages
import shutil
import numpy as np
import cv2


class GridWorld:

    _ACTION_SPACE_SIZE = 4
    _COLORS_RGB = {
        "WALL": (255, 0, 0),
        "GOAL": (0, 255, 0),
        "AGENT": (0, 0, 255)
    }
    MOVE_REWARD = -1.0
    MAX_FPS = 100       # <= 1000

    def __init__(self, side=8):
        self._side = side
        self.step_limit = 2 * (self._side ** 2)
        self.wall_reward = -self.step_limit
        self.oob_reward = -self.step_limit      # out of bounds
        self.goal_reward = self.step_limit

        self._wall = None
        self._goal = None
        self._agent = None

        self._step_count = None

        self._terminated = None
        self._truncated = None

    def reset(self):
        self._wall = np.random.randint(low=0, high=self._side, size=(2,), dtype=int)
        self._goal = np.random.randint(low=0, high=self._side, size=(2,), dtype=int)

        while np.all(self._wall == self._goal):
            self._goal = np.random.randint(low=0, high=self._side, size=(2,), dtype=int)

        self._agent = np.random.randint(low=0, high=self._side, size=(2,), dtype=int)

        while np.all(self._wall == self._agent) or np.all(self._goal == self._agent):
            self._agent = np.random.randint(low=0, high=self._side, size=(2,), dtype=int)

        self._step_count = 0
        self._terminated = False
        self._truncated = False

        return self._make_rgb_img(), "initial state"

    def step(self, action):

        assert self._agent is not None, "reset method should be called first to initialize the environment"
        assert action in list(range(self._ACTION_SPACE_SIZE)), f"action should be one of {list(range(self._ACTION_SPACE_SIZE))}"
        assert not self._terminated, "The episode was terminated. Call reset method to re-initialize."
        assert not self._truncated, "The episode timed out. Call reset method to re-initialize."

        self._step_count += 1

        reward = self.MOVE_REWARD
        info = "default"

        # clockwise from up
        if action == 0:
            # up
            self._agent += [-1, 0]
        elif action == 1:
            # right
            self._agent += [0, 1]
        elif action == 2:
            # down
            self._agent += [1, 0]
        elif action == 3:
            # left
            self._agent += [0, -1]

        # checking out of bounds
        if self._is_oob(self._agent):
            self._terminated = True
            reward = self.oob_reward
            info = "out of bounds"

        # checking step limit
        if self._step_count >= self.step_limit:
            self._truncated = True
            info = "step limit reached"

        # checking if the goal was achieved
        if np.all(self._agent == self._goal):
            reward = self.goal_reward
            self._terminated = True
            info = "goal achieved"

        # checking if the player hit the wall
        if np.all(self._agent == self._wall):
            reward = self.wall_reward
            self._terminated = True
            info = "hit the wall"

        return self._make_rgb_img(), reward, self._terminated, self._truncated, info

    def render(self):

        assert self._agent is not None, "reset method should be called first to initialize the environment"

        img_rgb = self._make_rgb_img()
        img_rgb = cv2.resize(img_rgb, (256, 256), interpolation=cv2.INTER_NEAREST)
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        cv2.imshow("GridWorld", img_bgr)
        cv2.waitKey(1000//self.MAX_FPS)

    def _make_rgb_img(self):
        img_rgb = np.zeros((self._side, self._side, 3), dtype=np.uint8)
        img_rgb[*self._wall] = self._COLORS_RGB["WALL"]
        img_rgb[*self._goal] = self._COLORS_RGB["GOAL"]
        if not self._is_oob(self._agent):
            img_rgb[*self._agent] = self._COLORS_RGB["AGENT"]
        return img_rgb

    def _is_oob(self, state):
        # out of bounds
        if (state[0] < 0) or (state[0] > self._side - 1):
            return True
        elif state[1] < 0 or state[1] > self._side-1:
            return True
        else:
            return False

    def __repr__(self):
        out_str = f"\nside: {self._side}"
        if self._agent is not None:
            out_str = f"{out_str}\nwall: {self._wall}\ngoal: {self._goal}\nagent: {self._agent}\n"
            terminal_size = shutil.get_terminal_size()
            if terminal_size.columns >= self._side:
                world = np.broadcast_to(np.array(["_"], dtype=str), shape=(self._side, self._side)).copy()
                world[*self._wall] = "W"
                world[*self._goal] = "G"
                world[*self._agent] = "A"
                for row in world:
                    row_str = "|".join(row)
                    out_str = f"{out_str}\n|{row_str}|"
        return out_str


if __name__ == "__main__":
    env = GridWorld()
    obs, _ = env.reset()
    print(env)

    terminated = False
    truncated = False

    while not (terminated or truncated):
        _, _, terminated, truncated, _ = env.step(np.random.randint(low=0, high=4, size=(1,), dtype=int))
        env.render()
