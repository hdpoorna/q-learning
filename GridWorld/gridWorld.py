"""
py311
hdpoorna
"""

# import packages
import time
from enum import Enum
import shutil
import numpy as np
import cv2


class WallRule(Enum):
    # when agent coincides
    TERMINATE = 0
    PENALIZE = 1
    THROUGH = 2     # do nothing


class GridWorld:

    _ACTION_SPACE_SIZE = 4
    _COLORS_RGB = {
        "WALL": (255, 0, 0),        # red
        "GOAL": (0, 255, 0),        # green
        "AGENT": (0, 0, 255)        # blue
    }
    MOVE_REWARD = -1.0
    MAX_FPS = 6       # <= 1000

    def __init__(self, side=8, oob_rule: WallRule = WallRule.THROUGH, wall_rule: WallRule = WallRule.THROUGH):

        assert side >= 2, "side >= 2"
        assert isinstance(oob_rule, WallRule), "oob_rule should be of type WallRule"
        assert isinstance(wall_rule, WallRule), "wall_rule should be of type WallRule"

        self._side = side
        self._oob_rule = oob_rule
        self._wall_rule = wall_rule

        self.step_limit = 2 * (self._side * 2)

        self.oob_reward = 0.0      # out of bounds
        self.wall_reward = 0.0
        self.goal_reward = 0.0

        self.set_rewards()

        self._wall = None
        self._goal = None
        self._agent = None

        self._step_count = None

        self._terminated = None
        self._truncated = None

    def set_rewards(self):

        if self._oob_rule == WallRule.TERMINATE:
            assert self._wall_rule == WallRule.TERMINATE, "oob severity is higher than wall"
            self.oob_reward = float(-self.step_limit)
            self.wall_reward = float(-self.step_limit)
            self.goal_reward = 2.0 * float(self.step_limit)
        elif self._oob_rule == WallRule.PENALIZE:
            assert self._wall_rule in [WallRule.TERMINATE, WallRule.PENALIZE], "oob severity is higher than wall"
            self.oob_reward = 0.1 * float(-self.step_limit)
            self.goal_reward = 2.0 * (-self.oob_reward) * self.step_limit
            if self._wall_rule == WallRule.TERMINATE:
                self.wall_reward = self.oob_reward * self.step_limit
            elif self._wall_rule == WallRule.PENALIZE:
                self.wall_reward = self.oob_reward
        else:
            # self.oob_reward = WallRule.THROUGH
            self.oob_reward = 0.0
            if self._wall_rule == WallRule.TERMINATE:
                self.wall_reward = float(-self.step_limit)
                self.goal_reward = 2.0 * float(self.step_limit)
            elif self._wall_rule == WallRule.PENALIZE:
                self.wall_reward = 0.1 * float(-self.step_limit)
                self.goal_reward = 2.0 * (-self.wall_reward) * self.step_limit
            else:
                # self._wall_rule = WallRule.THROUGH
                self.wall_reward = 0.0
                self.goal_reward = float(self.step_limit)

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

    def close(self):

        assert self._agent is not None, "reset method should be called first to initialize the environment"

        self._wall = None
        self._goal = None
        self._agent = None

        self._step_count = None

        self._terminated = None
        self._truncated = None

    def step(self, action):

        assert self._agent is not None, "reset method should be called first to initialize the environment"
        assert action in list(range(self._ACTION_SPACE_SIZE)), f"action should be one of {list(range(self._ACTION_SPACE_SIZE))}"
        assert not self._terminated, "The episode was terminated. Call reset method to re-initialize."
        assert not self._truncated, "The episode timed out. Call reset method to re-initialize."

        self._step_count += 1

        previous_state = self._agent.copy()

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
        is_oob, wrapped = self._is_oob(self._agent)
        if is_oob:
            if self._oob_rule == WallRule.TERMINATE:
                self._terminated = True
            elif self._oob_rule == WallRule.PENALIZE:
                self._agent = previous_state
            elif self._oob_rule == WallRule.THROUGH:
                self._agent = wrapped
            reward += self.oob_reward
            info = "out of bounds"

        # checking step limit
        if self._step_count >= self.step_limit:
            self._truncated = True
            info = "step limit reached"

        # checking if the goal was achieved
        if np.all(self._agent == self._goal):
            reward += self.goal_reward
            self._terminated = True
            info = "goal achieved"

        # checking if the player hit the wall
        if np.all(self._agent == self._wall):
            if self._wall_rule == WallRule.TERMINATE:
                self._terminated = True
            elif self._wall_rule == WallRule.PENALIZE:
                self._agent = previous_state
            elif self._wall_rule == WallRule.THROUGH:
                pass
            reward += self.wall_reward
            info = "hit the wall"

        return self._make_rgb_img(), reward, self._terminated, self._truncated, info

    def render(self):

        assert self._agent is not None, "reset method should be called first to initialize the environment"

        img_rgb = self._make_rgb_img()
        img_rgb = cv2.resize(img_rgb, (256, 256), interpolation=cv2.INTER_NEAREST)
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        cv2.imshow("GridWorld", img_bgr)
        cv2.waitKey(1000//self.MAX_FPS)
        return img_rgb

    def play(self):

        assert self._agent is not None, "reset method should be called first to initialize the environment"
        assert not self._terminated, "The episode was terminated. Call reset method to re-initialize."
        assert not self._truncated, "The episode timed out. Call reset method to re-initialize."

        info = None

        while True:

            img_rgb = self._make_rgb_img()
            img_rgb = cv2.resize(img_rgb, (256, 256), interpolation=cv2.INTER_NEAREST)
            img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
            cv2.imshow("GridWorld", img_bgr)
            k = cv2.waitKey(1000 // self.MAX_FPS)

            if self._terminated or self._truncated:
                print(info)
                break

            if k == 27:  # Esc key to stop
                break
            elif k == -1:  # normally -1 returned
                continue
            elif k == ord("w"):
                _, _, _, _, info = self.step(0)
            elif k == ord("d"):
                _, _, _, _, info = self.step(1)
            elif k == ord("s"):
                _, _, _, _, info = self.step(2)
            elif k == ord("a"):
                _, _, _, _, info = self.step(3)

    def _make_rgb_img(self):
        img_rgb = np.zeros((self._side, self._side, 3), dtype=np.uint8)
        if self._wall_rule != WallRule.THROUGH:
            img_rgb[tuple(self._wall)] = self._COLORS_RGB["WALL"]
        img_rgb[tuple(self._goal)] = self._COLORS_RGB["GOAL"]
        is_obb, _ = self._is_oob(self._agent)
        if not is_obb:
            img_rgb[tuple(self._agent)] = self._COLORS_RGB["AGENT"]
        return img_rgb

    def _is_oob(self, state):
        # out of bounds
        if state[0] < 0:
            wrapped = state + [self._side, 0]
            return True, wrapped
        elif state[0] > self._side-1:
            wrapped = state + [-self._side, 0]
            return True, wrapped
        elif state[1] < 0:
            wrapped = state + [0, self._side]
            return True, wrapped
        elif state[1] > self._side-1:
            wrapped = state + [0, -self._side]
            return True, wrapped
        else:
            return False, state

    def __repr__(self):
        out_str = f"\nside: {self._side}"
        if self._agent is not None:
            out_str = f"{out_str}\nwall: {self._wall}\ngoal: {self._goal}\nagent: {self._agent}\n"
            terminal_size = shutil.get_terminal_size()
            if terminal_size.columns >= self._side:
                world = np.broadcast_to(np.array(["_"], dtype=str), shape=(self._side, self._side)).copy()
                world[tuple(self._wall)] = "W"
                world[tuple(self._goal)] = "G"
                world[tuple(self._agent)] = "A"
                for row in world:
                    row_str = "|".join(row)
                    out_str = f"{out_str}\n|{row_str}|"
        return out_str


if __name__ == "__main__":
    env = GridWorld(wall_rule=WallRule.THROUGH)
    env.MAX_FPS = 6
    obs, _ = env.reset()
    print(env)

    MODE = "AUTO"

    if MODE == "PLAY":
        env.play()
    else:
        terminated = False
        truncated = False

        while not (terminated or truncated):
            _, _, terminated, truncated, _ = env.step(np.random.randint(low=0, high=env._ACTION_SPACE_SIZE, size=(1,), dtype=int))
            env.render()

    time.sleep(0.5)
    cv2.destroyAllWindows()
    env.close()
