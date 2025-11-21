import torch # type: ignore
import gymnasium # type: ignore
import gym_environment

from stable_baselines3 import PPO # type: ignore
from sb3_contrib import RecurrentPPO, TRPO # type: ignore

import numpy as np # type: ignore
import cv2 # type: ignore

if __name__ == '__main__':
    model_type = "TRPO"
    render_fps = 4
    grid_size = (8, 6)
    window_size = (800, 600)
    testing_episode_count = 100
    training_timesteps = 500000

    env = gymnasium.make(
        'gym_environment/Snake-v0', 
        render_mode="rgb_array", 
        grid_size=grid_size, 
        window_size=window_size
        )
    
    if model_type == "RPPO":
        model = RecurrentPPO('MultiInputLstmPolicy', env, verbose=1, ent_coef=0.05)     
    elif model_type == "PPO":
        model = PPO('MultiInputPolicy', env, verbose=1, ent_coef=0.05)
    elif model_type == "TRPO":
        model = TRPO('MultiInputPolicy', env, verbose=1)
    vec_env = model.get_env()

    model.learn(total_timesteps=training_timesteps)

    # do testing episodes and record video
    # fourcc = cv2.VideoWriter_fourcc(*'XVID')
    # out = cv2.VideoWriter('output.avi', fourcc, render_fps, (800, 600))

    cv2.startWindowThread()
    cv2.namedWindow("game")

    for _ in range(testing_episode_count):
        observation = vec_env.reset()
        terminated = False
        steps = 0
        reward = 0

        lstm_states = None
        episode_starts = np.ones((1,), dtype=bool)
        while not terminated:    
            if model_type == "RPPO":
                action, lstm_states = model.predict(observation, state=lstm_states, episode_start=episode_starts, deterministic=True)
            elif model_type == "PPO" or model_type == "TRPO":
                action, _states = model.predict(observation, deterministic=True)

            observation, reward_, terminated, _, info = env.step(action)
            steps += 1
            reward += reward_

            if not terminated:
                img = env.render()
                # out.write(img)
                cv2.imshow("game", env.render())
                cv2.waitKey(int(1000 * 1 / render_fps))

    cv2.destroyAllWindows()
    # out.release()