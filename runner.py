import sys
import os

import gym_environment

from stable_baselines3 import PPO  # type: ignore
from stable_baselines3.common.env_util import make_vec_env  # type: ignore
from stable_baselines3.common.vec_env import SubprocVecEnv  # type: ignore
from sb3_contrib import RecurrentPPO, TRPO, MaskablePPO  # type: ignore
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy  # type: ignore
from sb3_contrib.common.maskable.utils import get_action_masks  # type: ignore

import numpy as np  # type: ignore
import cv2  # type: ignore

if __name__ == "__main__":
    tb_log_path = "/home/docker_user/logs/"
    env_id = "gym_environment/Snake-v0"

    # ===============Edit below=================
    render_fps = 16
    grid_size = (5, 5)
    window_size = (800, 600)
    testing_episode_count = int(1)
    training_timesteps = int(1e6)
    n_envs = 8

    load_pretrained = True
    model_type = "MPPO"  # default maskable ppo
    
    ent_coef=0.001 # sb default is 0
    # changes made below here only apply to maskable PPO and PPO
    learning_rate = 0.0003
    n_steps = 2048
    batch_size = 64
    n_epochs = 10
    gamma = 0.99
    gae_lambda = 0.95
    clip_range = 0.2
    # ===============Edit above=================

    os.system(
        f"tensorboard --host 0.0.0.0 --port 6006 --logdir {tb_log_path} &"
    )  # a bit hacky

    timestep_start = 0
    window_size = (
        window_size[0] // int(np.sqrt(n_envs)),
        window_size[1] // int(np.sqrt(n_envs)),
    )

    if not os.path.exists("/home/docker_user/model/"):
        os.makedirs("/home/docker_user/model/")

    if os.path.exists("./model/"):
        for file in os.listdir("./model/"):
            if (
                file.startswith(f"{str(grid_size[0])}x{str(grid_size[1])}")
                and file.split("_")[-1] == f"{model_type}.zip"
            ):
                ts_ = int(file.split("_")[-2])
                if ts_ >= timestep_start:
                    timestep_start = ts_

    model_filename = f"./model/{str(grid_size[0])}x{str(grid_size[1])}_{str(timestep_start)}_{model_type}.zip"

    env = make_vec_env(
        env_id,
        n_envs=n_envs,
        env_kwargs=dict(
            grid_size=grid_size,
            window_size=window_size,
        ),
        vec_env_cls=SubprocVecEnv,
    )

    if model_type == "RPPO":
        if load_pretrained and os.path.exists(model_filename):
            print(f"Loading model {model_filename}")
            model = RecurrentPPO.load(
                model_filename,
                env=env,
                verbose=1,
                device="cpu",
                ent_coef=ent_coef,
                tensorboard_log=tb_log_path
            )
            timestep_start = int(model_filename.split("_")[-2])
        else:
            model = RecurrentPPO(
                "MlpLstmPolicy",
                env,
                verbose=1,
                device="cpu",
                ent_coef=ent_coef,
                tensorboard_log=tb_log_path
            )
    elif model_type == "PPO":
        if load_pretrained and os.path.exists(model_filename):
            print(f"Loading model {model_filename}")
            model = PPO.load(
                model_filename,
                env=env,
                verbose=1,
                device="cpu",
                ent_coef=ent_coef,
                tensorboard_log=tb_log_path,
                learning_rate=learning_rate,
                n_steps=n_steps,
                batch_size=batch_size,
                n_epochs=n_epochs,
                gamma=gamma,
                gae_lambda=gae_lambda,
                clip_range=clip_range,
            )
            timestep_start = int(model_filename.split("_")[-2])
        else:
            model = PPO(
                "MlpPolicy",
                env,
                verbose=1,
                device="cpu",
                ent_coef=ent_coef,
                tensorboard_log=tb_log_path,
                learning_rate=learning_rate,
                n_steps=n_steps,
                batch_size=batch_size,
                n_epochs=n_epochs,
                gamma=gamma,
                gae_lambda=gae_lambda,
                clip_range=clip_range,
            )
    elif model_type == "TRPO":
        if load_pretrained and os.path.exists(model_filename):
            print(f"Loading model {model_filename}")
            model = TRPO.load(
                model_filename,
                env=env,
                verbose=1,
                device="cpu",
                tensorboard_log=tb_log_path
            )
            timestep_start = int(model_filename.split("_")[-2])
        else:
            model = TRPO(
                "MlpPolicy",
                env,
                verbose=1,
                device="cpu",
                tensorboard_log=tb_log_path
            )

    else:
        if load_pretrained and os.path.isfile(model_filename):
            print(f"Loading model {model_filename}")
            model = MaskablePPO.load(
                model_filename,
                env=env,
                verbose=1,
                device="cpu",
                ent_coef=ent_coef,
                tensorboard_log=tb_log_path,
                learning_rate=learning_rate,
                n_steps=n_steps,
                batch_size=batch_size,
                n_epochs=n_epochs,
                gamma=gamma,
                gae_lambda=gae_lambda,
                clip_range=clip_range,
            )
            timestep_start = int(model_filename.split("_")[-2])
        else:
            model = MaskablePPO(
                MaskableActorCriticPolicy,
                env,
                verbose=1,
                device="cpu",
                ent_coef=ent_coef,
                tensorboard_log=tb_log_path,
                learning_rate=learning_rate,
                n_steps=n_steps,
                batch_size=batch_size,
                n_epochs=n_epochs,
                gamma=gamma,
                gae_lambda=gae_lambda,
                clip_range=clip_range,
            )

    model.learn(total_timesteps=training_timesteps)

    model_filename = f"/home/docker_user/model/{str(grid_size[0])}x{str(grid_size[1])}_{str(training_timesteps + timestep_start)}_{model_type}.zip"
    model.save(model_filename)

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
                actions, lstm_states = model.predict(
                    observations,
                    state=lstm_states,
                    episode_start=episode_starts,
                    deterministic=True,
                )
            elif model_type == "PPO" or model_type == "TRPO":
                actions, _states = model.predict(observations, deterministic=True)
            else:
                action_masks = get_action_masks(env)
                actions, _states = model.predict(
                    observations, action_masks=action_masks, deterministic=True
                )

            observations, rewards_, terminated_, infos = env.step(actions)
            steps += 1

            rewards += rewards_

            if not np.all(terminated):
                img = env.render()
                cv2.imshow("game", env.render())
                cv2.waitKey(int(1000 * 1 / render_fps))

    cv2.destroyAllWindows()
