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
                "target": spaces.MultiDiscrete([self.grid_size[0], self.grid_size[1]], dtype=int),
                "left_distance": spaces.Discrete(self.grid_size[0], dtype=int),
                "right_distance": spaces.Discrete(self.grid_size[0], dtype=int),
                "up_distance": spaces.Discrete(self.grid_size[1], dtype=int),
                "down_distance": spaces.Discrete(self.grid_size[1], dtype=int),                
                #"grid": spaces.Box(-1, 1, (*self.grid_size, 2), dtype=int)
            }
        )        

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self._agent_location = self.np_random.integers(0, self.grid_size, size=2, dtype=int)

        self._target_location = self._agent_location
        while np.array_equal(self._target_location, self._agent_location):
            self._target_location = self.np_random.integers(0, self.grid_size, size=2, dtype=int)

        self.v_tiles = np.zeros((*self.grid_size, 2), dtype=int) # represent snake tiles as the direction they move in
        self.v_tiles[self._agent_location[0], self._agent_location[1]] = self._action_to_direction[Actions.UP]

        observation = self._get_obs()
        info = self._get_info()

        return observation, info

    def step(self, action):
        reward = -0.1
        terminated = False

        direction = self._action_to_direction[Actions(action)]

        if np.array_equal(direction, -self._last_direction): # do nothing when illegal input pressed
            direction = -direction
        v_tiles_next = np.zeros(self.v_tiles.shape, dtype=int)
        
        self.v_tiles[self._agent_location[0], self._agent_location[1]] = direction       
      
        if np.array_equal(self._agent_location + direction, self._target_location):
            v_tiles_next = self.v_tiles
            while np.linalg.norm(self._target_location - (self._agent_location + direction)) < 2:
                self._target_location = self.np_random.integers(0, self.grid_size, size=2, dtype=int)
            reward = 100

        for x in range(self.v_tiles.shape[0]):
            for y in range(self.v_tiles.shape[1]):
                if np.array_equal(self.v_tiles[x, y], (0, 0)):
                    continue
                tile_dir = self.v_tiles[x, y]
                next_tile = np.array([x + tile_dir[0], y + tile_dir[1]])
                if np.any(next_tile < 0) or np.any(next_tile >= self.grid_size):
                    terminated = True
                    break
                else:
                    v_tiles_next[next_tile[0], next_tile[1]] = self.v_tiles[next_tile[0], next_tile[1]]

        self._agent_location += direction
        terminated = terminated \
            or np.any(self._agent_location < 0) or np.any(self._agent_location >= self.grid_size) \
            or not np.array_equal(self.v_tiles[self._agent_location[0], self._agent_location[1]], (0, 0)) 
        
        if not terminated:
            v_tiles_next[self._agent_location[0], self._agent_location[1]] = direction
            self.v_tiles = v_tiles_next
        else:
            v_tiles_next[self._agent_location[0] - direction[0], self._agent_location[1] - direction[1]] = np.array([0, 0])
            self.v_tiles = v_tiles_next

        self._last_direction = direction
        reward = reward if not terminated else -100
        observation = self._get_obs()
        info = self._get_info()
        return observation, reward, terminated, False, info
    
    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()

    def _get_obs(self):
        up_distance = self._agent_location[1]
        down_distance = self.grid_size[1] - self._agent_location[1] - 1
        left_distance = self._agent_location[0]
        right_distance = self.grid_size[0] - self._agent_location[0] - 1

        if up_distance > 1 and self._agent_location[0] < self.grid_size[0]:
            for y in range(1, up_distance):
                if not np.array_equal(self.v_tiles[self._agent_location[0], self._agent_location[1] - y], (0, 0)):
                    up_distance = self._agent_location[1] - y
                    break
        
        if down_distance > 1 and self._agent_location[0] < self.grid_size[0]:
            for y in range(1, down_distance):
                if not np.array_equal(self.v_tiles[self._agent_location[0], self._agent_location[1] + y], (0, 0)):
                    down_distance = self._agent_location[1] + y
                    break

        if left_distance > 1 and self._agent_location[1] < self.grid_size[1]:
            for x in range(1, left_distance):
                if not np.array_equal(self.v_tiles[self._agent_location[0] - x, self._agent_location[1]], (0, 0)):
                    left_distance = self._agent_location[0] - x
                    break
    
        if right_distance > 1 and self._agent_location[1] < self.grid_size[1]:
            for x in range(1, right_distance):
                if not np.array_equal(self.v_tiles[self._agent_location[0] + x, self._agent_location[1]], (0, 0)):
                    right_distance = self._agent_location[0] + x
                    break

        down_distance = down_distance if down_distance >= 0 else 0
        up_distance = up_distance if up_distance >= 0 else 0
        left_distance = down_distance if left_distance >= 0 else 0
        right_distance = right_distance if right_distance >= 0 else 0

        return {
            "agent": self._agent_location, 
            "target": self._target_location,
            "left_distance": left_distance,
            "right_distance": right_distance,
            "up_distance": up_distance,
            "down_distance": down_distance,
            #"grid": self.v_tiles
            }

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