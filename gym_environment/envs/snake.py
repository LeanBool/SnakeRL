import sys, os
sys.stdout = open(os.devnull,'w') # suppress pygame import warning
sys.stderr = open(os.devnull,'w')
import pygame
sys.stdout = sys.__stdout__
sys.stderr = sys.__stdout__

from enum import Enum
import numpy as np
import gymnasium as gym
from gymnasium import spaces

class Actions(Enum):
    RIGHT = 0
    UP = 1
    LEFT = 2
    DOWN = 3

class SnakeEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode=None, window_size=None, grid_size=None):
        self.grid_size = (6, 4) if not grid_size else grid_size
        self.render_mode = render_mode
        self.window_size = (800, 600) if not window_size else window_size

        self._agent_location = np.array([-1, -1], dtype=int)
        self._target_location = np.array([-1, -1], dtype=int)

        self.action_space = spaces.Discrete(4)
        self._action_to_direction = {
            Actions.RIGHT.value: np.array([1, 0]),
            Actions.UP.value: np.array([0, 1]),
            Actions.LEFT.value: np.array([-1, 0]),
            Actions.DOWN.value: np.array([0, -1]),
        }

        self.observation_space = spaces.Dict(
            {
                "agent": spaces.MultiDiscrete([self.grid_size[0], self.grid_size[1]], dtype=int),
                "target": spaces.MultiDiscrete([self.grid_size[0], self.grid_size[1]], dtype=int),
            }
        )

        self.window = None

    def reset(self):
        pass

    def step(self):
        pass

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