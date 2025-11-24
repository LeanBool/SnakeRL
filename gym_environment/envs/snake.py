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

    _collected_target = False
    _ticks_since_last_collect = 0
    _max_ticks_since_last_collect = 50
    _last_min_dist = 0
    _score = 0

    _obs_vec = None

    def __init__(self, window_size=None, grid_size=None):
        self.window = None
        self.clock = None

        self.grid_size = (12, 6) if not grid_size else grid_size
        self.render_mode = "rgb_array"
        self.window_size = (800, 600) if not window_size else window_size

        ratio_window = self.window_size[0] / self.window_size[1]
        ratio_grid = self.grid_size[0] / self.grid_size[1]
        if not np.isclose(ratio_grid, ratio_window): # change window shape if the screen ratio isnt the same as grid ratio
            self.window_size = (self.window_size[0] * ratio_grid / ratio_window, self.window_size[1])            

        self._agent_location = np.array([0, 0], dtype=int)
        self._target_location = np.array([0, 0], dtype=int)
        self._last_direction = self._action_to_direction[self.np_random.choice(Actions)]

        self.action_space = spaces.Discrete(4)

        # each tile is either: empty tile, target, snake head, snake tile (4 dirs)
        # self.observation_space = spaces.MultiDiscrete([*(7 * np.ones(np.prod(self.grid_size)))], dtype=int)

        self._obs_vec = np.zeros(np.prod(self.grid_size))
        v = 2 * np.ones(np.prod(self.grid_size))
        self.observation_space = spaces.MultiDiscrete([*self.grid_size, *self.grid_size, *v], dtype=int)
        self._max_ticks_since_last_collect = np.prod(self.grid_size) * 2

        # self.observation_space = spaces.Dict(
        #     {
        #         "agent": spaces.MultiDiscrete([self.grid_size[0], self.grid_size[1]], dtype=int),
        #         "target": spaces.MultiDiscrete([self.grid_size[0], self.grid_size[1]], dtype=int),
        #         # "left_distance": spaces.Discrete(self.grid_size[0], dtype=int),
        #         # "right_distance": spaces.Discrete(self.grid_size[0], dtype=int),
        #         # "up_distance": spaces.Discrete(self.grid_size[1], dtype=int),
        #         # "down_distance": spaces.Discrete(self.grid_size[1], dtype=int),                
        #         # "grid": spaces.Box(-1, 1, (*self.grid_size, 2), dtype=int),
        #         # "grid": spaces.MultiDiscrete([3 for _ in range(int(2*np.prod(self.grid_size)))], dtype=int),
        #     }
        # )        

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self._agent_location = self.np_random.integers(0, self.grid_size, size=2, dtype=int)

        self._target_location = self._agent_location
        while np.array_equal(self._target_location, self._agent_location):
            self._target_location = self.np_random.integers(0, self.grid_size, size=2, dtype=int)

        self.v_tiles = np.zeros((*self.grid_size, 2), dtype=int) # represent snake tiles as the direction they move in
        self.v_tiles[self._agent_location[0], self._agent_location[1]] = self._action_to_direction[self.np_random.choice(Actions)]

        self._collected_target = False
        self._ticks_since_last_collect = 0
        self._max_ticks_since_last_collect = 50
        self._last_min_dist = np.prod(self.grid_size)
        self._score = 0        

        observation = self._get_obs()
        info = self._get_info()
    
        return observation, info

    def step(self, action):
        terminated = False

        self._max_ticks_since_last_collect = int(np.prod(self.grid_size) + (self._score + 1))

        direction = self._action_to_direction[Actions(action)]
        if np.array_equal(direction, -self._last_direction): # do nothing when illegal input pressed
            direction = -direction
        
        v_tiles_next = np.zeros(self.v_tiles.shape, dtype=int)
        self.v_tiles[self._agent_location[0], self._agent_location[1]] = direction       
      
        if np.array_equal(self._agent_location + direction, self._target_location):
            v_tiles_next = self.v_tiles
            self._target_location = self.np_random.integers(0, self.grid_size, size=2, dtype=int)
            while not np.array_equal(self.v_tiles[self._target_location[0], self._target_location[1]], (0, 0)):
                self._target_location = self.np_random.integers(0, self.grid_size, size=2, dtype=int)
            self._collected_target = True
            self._ticks_since_last_collect = 0
            self._score += 1
        else:
            self._collected_target = False
            self._ticks_since_last_collect += 1

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
            or not np.array_equal(self.v_tiles[self._agent_location[0], self._agent_location[1]], (0, 0)) \
            or self._ticks_since_last_collect > self._max_ticks_since_last_collect 
                
        if not terminated:
            v_tiles_next[self._agent_location[0], self._agent_location[1]] = direction
            self.v_tiles = v_tiles_next
        else:
            v_tiles_next[self._agent_location[0] - direction[0], self._agent_location[1] - direction[1]] = np.array([0, 0])
            self.v_tiles = v_tiles_next
    
        self._last_direction = direction
        reward = self._get_reward(terminated)
        observation = self._get_obs()
        info = self._get_info()

        if self._score >= np.prod(self.grid_size) - 1:
            reward += 10*np.prod(self.grid_size)
            terminated = True

        if not self._collected_target:
            self._last_min_dist = min(self._last_min_dist, np.linalg.norm(self._agent_location - self._target_location, ord=1))
        else:
            self._last_min_dist = np.prod(self.grid_size)

        return observation, reward, terminated, False, info
    
    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()

    def _get_obs(self):
        self._obs_vec = np.zeros(4 + np.prod(self.grid_size), dtype=int)

        for x in range(self.v_tiles.shape[0]):
            for y in range(self.v_tiles.shape[1]):                
                if str(self.v_tiles[x, y]) == "[0 0]":
                    self._obs_vec[x*self.grid_size[1] + y + 4] = 0
                else:
                    self._obs_vec[x*self.grid_size[1] + y + 4] = 1
        
        _clipped_agent = np.clip(self._agent_location, np.zeros(2), np.array(self.grid_size) - 1)
        _clipped_target = np.clip(self._target_location, np.zeros(2), np.array(self.grid_size) - 1)

        self._obs_vec[0] = _clipped_agent[0]
        self._obs_vec[1] = _clipped_agent[1]
        self._obs_vec[2] = _clipped_target[0]
        self._obs_vec[3] = _clipped_target[1]

        return self._obs_vec

    def _get_reward(self, terminated = False, direction = None):
        reward = 0

        if terminated:
            if self._ticks_since_last_collect > self._max_ticks_since_last_collect:
                return -np.prod(self.grid_size) * 4 * self._score
            return -np.prod(self.grid_size) * 2
        
        if self._collected_target:
            reward += np.prod(self.grid_size) * 2 * self._score                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       
        
        # encourage moving quickly towards target especially early on
        if self._score < np.prod(self.grid_size) // 2 and self._last_min_dist > np.linalg.norm(self._agent_location - self._target_location, ord=1):
            reward += 1.5*(np.prod(self.grid_size) - self._score)
        elif self._score < np.prod(self.grid_size) // 2 and self._last_min_dist < np.linalg.norm(self._agent_location - self._target_location, ord=1):
            reward -= 0.1*(np.prod(self.grid_size) - self._score)

        return reward

    def _get_info(self):
        return {
            "distance": np.linalg.norm(self._agent_location - self._target_location, ord=1), # manhattan distance
            "score": self._score,            
        }
    
    def _render_frame(self):        
        canvas = pygame.Surface(self.window_size)
        canvas.fill((255, 255, 255))

        cell_size = min(self.window_size[0] // self.grid_size[0], self.window_size[1] // self.grid_size[1])

        pygame.draw.rect(
            canvas,
            (0, 0, 0),
            pygame.Rect(
                (0, 0),
                self.window_size,
            ),
            1
        )

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