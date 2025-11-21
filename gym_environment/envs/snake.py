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
    metadata = {"render_modes": ["rgb_array"], "render_fps": 4}
    _action_to_direction = {
            Actions.RIGHT: np.array([1, 0]),
            Actions.UP: np.array([0, 1]),
            Actions.LEFT: np.array([-1, 0]),
            Actions.DOWN: np.array([0, -1]),
    }

    def __init__(self, render_mode=None, window_size=None, grid_size=None):
        self.window = None
        self.clock = None

        self.grid_size = (12, 6) if not grid_size else grid_size
        self.render_mode = render_mode
        self.window_size = (800, 600) if not window_size else window_size

        self._agent_location = np.array([-1, -1], dtype=int)
        self._target_location = np.array([-1, -1], dtype=int)
        self._last_direction = self._action_to_direction[self.np_random.choice(Actions)]

        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Dict(
            {
                "agent": spaces.MultiDiscrete([self.grid_size[0], self.grid_size[1]], dtype=int),
                "target": spaces.MultiDiscrete([self.grid_size[0], self.grid_size[1]], dtype=int)
            }
        )        

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self._agent_location = np.array((max(0, self.grid_size[0]//2 - 1), max(0, self.grid_size[1]//2 - 1))) # spawn in center

        self._target_location = self._agent_location
        while np.array_equal(self._target_location, self._agent_location):
            self._target_location = self.np_random.integers(0, self.grid_size, size=2, dtype=int)

        self.v_tiles = np.zeros((*self.grid_size, 2), dtype=int) # represent snake tiles as the direction they move in
        self.v_tiles[self._agent_location[0], self._agent_location[1]] = self._action_to_direction[Actions.UP]

        observation = self._get_obs()
        info = self._get_info()

        return observation, info

    def step(self, action):
        reward = 0
        terminated = False

        direction = self._action_to_direction[Actions(action)]

        if np.array_equal(direction, -self._last_direction): # do nothing when illegal input pressed
            direction = -direction
        v_tiles_next = np.zeros(self.v_tiles.shape, dtype=int)
        
        self.v_tiles[self._agent_location[0], self._agent_location[1]] = direction       
      
        if np.array_equal(self._agent_location + direction, self._target_location):
            v_tiles_next = self.v_tiles
            while np.array_equal(self._target_location, self._agent_location + direction):
                self._target_location = self.np_random.integers(0, self.grid_size, size=2, dtype=int)

            reward = 100

        for x in range(self.v_tiles.shape[0]):
            for y in range(self.v_tiles.shape[1]):
                if np.array_equal(self.v_tiles[x, y], (0, 0)):
                    continue
                tile_dir = self.v_tiles[x, y]
                try:
                    v_tiles_next[x + tile_dir[0], y + tile_dir[1]] = self.v_tiles[x + tile_dir[0], y + tile_dir[1]]
                except:
                    terminated = True
                    break

        self._agent_location += direction
        terminated = terminated or \
            np.any(self._agent_location < 0) or \
            self._agent_location[0] >= self.grid_size[0] or self._agent_location[1] >= self.grid_size[1] or \
            not np.array_equal(self.v_tiles[self._agent_location[0], self._agent_location[1]], (0, 0)) \
            # or np.array_equal(self._last_direction, -direction)
        reward = reward if not terminated else 0
        observation = self._get_obs()
        info = self._get_info()

        if not terminated:
            v_tiles_next[self._agent_location[0], self._agent_location[1]] = direction
            self.v_tiles = v_tiles_next
        else:
            v_tiles_next[self._agent_location[0] - direction[0], self._agent_location[1] - direction[1]] = np.array([0, 0])
            self.v_tiles = v_tiles_next

        self._last_direction = direction
        return observation, reward, terminated, False, info
    
    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()

    def _get_obs(self):
        return {"agent": self._agent_location, "target": self._target_location}

    def _get_info(self):
        return {
            "distance": np.linalg.norm(self._agent_location - self._target_location, ord=1) # manhattan distance
        }
    
    def _render_frame(self):        
        canvas = pygame.Surface(self.window_size)
        canvas.fill((255, 255, 255))

        cell_size = min(self.window_size[0] // self.grid_size[0], self.window_size[1] // self.grid_size[1])

        pygame.draw.circle(
            canvas, 
            (255, 0, 0), 
            (self._target_location + 0.5)*cell_size, 
            cell_size / 4
        )

        pygame.draw.rect(
            canvas,
            (0, 0, 0),
            pygame.Rect(
                self._agent_location * cell_size,
                (cell_size , cell_size),
            )
        )

        for i in range(self.v_tiles.shape[0]):
            for j in range(self.v_tiles.shape[1]):
                if not np.array_equal(self.v_tiles[i, j], (0, 0)):
                    pygame.draw.rect(
                        canvas,
                        (0, 0, 0),
                        pygame.Rect(
                            np.array([i, j]) * cell_size,
                            (cell_size , cell_size),
                        )
                    )

        return np.transpose(
            np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
        )
            
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