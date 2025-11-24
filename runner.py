import sys
import os

import torch # type: ignore
import gymnasium # type: ignore
import gym_environment

from stable_baselines3 import PPO # type: ignore
from stable_baselines3.common.env_util import make_vec_env # type: ignore
from stable_baselines3.common.vec_env import SubprocVecEnv # type: ignore
from sb3_contrib import RecurrentPPO, TRPO # type: ignore

import numpy as np # type: ignore
import cv2 # type: ignore

from time import time

if __name__ == '__main__':
    env_id = "gym_environment/Snake-v0"
    model_type = "PPO"
    render_fps = 4
    grid_size = (4, 4)
    window_size = (800, 600)
    testing_episode_count = int(1e4)
    training_timesteps = int(5e5)
    n_envs = 8
    window_size = (window_size[0] // int(np.sqrt(n_envs)), window_size[1] // int(np.sqrt(n_envs)))

    env = make_vec_env(
        env_id, 
        n_envs=n_envs,
        env_kwargs = dict( 
            grid_size=grid_size, 
            window_size=window_size,
        ),
        vec_env_cls=SubprocVecEnv
    )

    if model_type == "RPPO":
        model = RecurrentPPO('MlpLstmPolicy', env, verbose=1, device='cpu', ent_coef=0.01)     
    elif model_type == "PPO":
        model = PPO('MlpPolicy', env, verbose=1, device='cpu', ent_coef=0.01)
    elif model_type == "TRPO":
        model = TRPO('MlpPolicy', env, verbose=1, device='cpu')    
    model.learn(total_timesteps=training_timesteps)    
    
    cv2.startWindowThread()
    cv2.namedWindow("game")

    observations = model.get_env().reset()
    for _ in range(testing_episode_count):
        observations = env.reset()
        terminated = np.zeros(n_envs, dtype=bool)
        steps = np.zeros(n_envs, dtype=int)
        rewards = np.zeros(n_envs)
        lstm_states = None
        episode_starts = np.ones((n_envs,), dtype=bool)

        while not np.all(terminated):
            if model_type == "RPPO":
                actions, lstm_states = model.predict(observations, state=lstm_states, episode_start=episode_starts, deterministic=True)
            else:
                actions, _states = model.predict(observations, deterministic=True)

            observations, rewards_, terminated_, infos = env.step(actions)
            steps += 1

            rewards += rewards_

            if not np.all(terminated):
                img = env.render()
                cv2.imshow("game", env.render())
                cv2.waitKey(int(1000 * 1 / render_fps))


    cv2.destroyAllWindows()