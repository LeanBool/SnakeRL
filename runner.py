import torch # type: ignore
import gymnasium # type: ignore
import gym_environment

from stable_baselines3 import PPO # type: ignore
from stable_baselines3.common.evaluation import evaluate_policy # type: ignore

import cv2 # type: ignore

if __name__ == '__main__':
    render_fps = 4
    testing_episode_count = 100
    training_timesteps = 500000

    env = gymnasium.make('gym_environment/Snake-v0', render_mode="rgb_array")
    model = PPO('MultiInputPolicy', env, verbose=1)
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
        while not terminated:
            action, _state = model.predict(observation, deterministic=True)
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