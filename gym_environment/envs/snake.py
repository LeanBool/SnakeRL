import sys, os
sys.stdout = open(os.devnull,'w') # suppress pygame import warning about deprecated dependency
sys.stderr = open(os.devnull,'w')
import pygame # type: ignore
sys.stdout = sys.__stdout__
sys.stderr = sys.__stdout__

from enum import Enum
import numpy as np # type: ignore
import gymnasium as gym # type: ignore
from gymnasium import spaces # type: ignore

class Actions(Enum):
    RIGHT = 0
    UP = 1
    LEFT = 2
    DOWN = 3

class SnakeEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}
    _action_to_direction = {
            Actions.RIGHT: np.array([1, 0]),
            Actions.UP: np.array([0, 1]),
            Actions.LEFT: np.array([-1, 0]),
            Actions.DOWN: np.array([0, -1]),
    }

    def __init__(self, render_mode=None, window_size=None, grid_size=None):
        self.window = None
        self.clock = None

        self.grid_size = (6, 4) if not grid_size else grid_size
        self.render_mode = render_mode
        self.window_size = (800, 600) if not window_size else window_size

        self._agent_location = np.array([-1, -1], dtype=int)
        self._target_location = np.array([-1, -1], dtype=int)

        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Dict(
            {
                "agent": spaces.MultiDiscrete([self.grid_size[0], self.grid_size[1]], dtype=int),
                "target": spaces.MultiDiscrete([self.grid_size[0], self.grid_size[1]], dtype=int),
            }
        )

        self.v_tiles = np.zeros((*self.grid_size, 2), dtype=int)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self._agent_location = (max(0, self.grid_size[0]//2 - 1), max(0, self.grid_size[1]//2 - 1)) # spawn in center

        self._target_location = self._agent_location
        while np.array_equal(self._target_location, self._agent_location):
            self._target_location = self.np_random.integers(0, self.grid_size, size=2, dtype=int)

        self.v_tiles[self._agent_location] = self._action_to_direction[Actions.UP]

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info

    def step(self, action):
        reward = 1

        direction = self._action_to_direction[Actions(action)]
        v_tiles_next = np.zeros(self.v_tiles.shape, dtype=int)

        if not np.array_equal(self._agent_location + direction, self._target_location):
            for x in range(self.v_tiles.shape[0]):
                for y in range(self.v_tiles.shape[1]):
                    if np.array_equal(self.v_tiles[x, y], (0, 0)) or np.array_equal((x, y), self._agent_location):
                        continue
                    tile_dir = self.v_tiles[x, y]
                    v_tiles_next[x + tile_dir[0], y + tile_dir[1]] = tile_dir
        else:
            v_tiles_next[self._agent_location[0], self._agent_location[1]] = direction
            reward = 100

        self._agent_location += direction
        terminated = np.any(self._agent_location < 0) or self._agent_location[0] >= self.grid_size[0] or self._agent_location[1] >= self.grid_size[1] \
            or not np.array_equal(v_tiles_next[self._agent_location[0], self._agent_location[1]], (0, 0))
        reward = reward if not terminated else 0
        observation = self._get_obs()
        info = self._get_info()

        if not terminated:
            v_tiles_next[self._agent_location[0], self._agent_location[1]] = direction
            self.v_tiles = v_tiles_next

        return observation, reward, terminated, False, info

    def _get_obs(self):
        return {"agent": self._agent_location, "target": self._target_location}

    def _get_info(self):
        return {
            "distance": np.linalg.norm(self._agent_location - self._target_location, ord=1) # manhattan distance
        }

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()
    
    def _render_frame(self):
        if self.window and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(self.window_size)

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()

    def _print_grid(self):
        out = ""

        for x in range(self.grid_size[0]):
            for y in range(self.grid_size[1]):
                if np.array_equal(self._agent_location, (x, y)):
                    out += "o"
                elif not np.array_equal(self.v_tiles[x, y], (0, 0)):
                    out += "+"
                elif np.array_equal(self._target_location, (x, y)):
                    out += "x"
                else:
                    out += "#"
            out += "\n"
        
        print(out)