import os
import gym_environment
import torch  # type: ignore
from stable_baselines3.common.env_util import make_vec_env  # type: ignore
from stable_baselines3.common.vec_env import SubprocVecEnv  # type: ignore
from stable_baselines3.common.callbacks import BaseCallback  # type: ignore
from sb3_contrib import MaskablePPO  # type: ignore
from sb3_contrib.common.maskable.utils import get_action_masks  # type: ignore

import numpy as np  # type: ignore
import cv2  # type: ignore


class CurriculumLearningCallback(BaseCallback):
    def __init__(
                    self,
                    env_id="gym_environment/Snake-v0",
                    device=None,
                    model_type="MPPO",
                    ent_coef=0.001,
                    n_envs=8,
                    tensorboard_log="/home/docker_user/logs/",
                    learning_rate=0.0003,
                    n_steps=2048,
                    batch_size=64,
                    n_epochs=10,
                    gamma=0.99,
                    gae_lambda=0.95,
                    clip_range=0.2,
                    stage=0,
                    curriculum_transitions=None,
                ):

        super().__init__()

        self._curriculum_transitions = curriculum_transitions
        self._stage = stage
        self._env_id = env_id
        self._model_type = model_type
        self._n_envs = n_envs
        self._device = device
        self._ent_coef = ent_coef
        self._tensorboard_log = tensorboard_log
        self._learning_rate = learning_rate
        self._n_steps = n_steps
        self._batch_size = batch_size
        self._n_epochs = n_epochs
        self._gamma = gamma
        self._gae_lambda = gae_lambda
        self._clip_range = clip_range

    def _on_step(self):
        next_transition = self._curriculum_transitions[self._stage][0]
        if self._stage < len(self._curriculum_transitions) and  \
                self.num_timesteps >= next_transition:

            print(f"Finished stage {self._stage}.")
            file_name = f"/home/docker_user/model/{self._stage}"
            self.model.save(file_name)

            del self.model
            return False

        return True


def get_model_filename(grid_size, model_type="MPPO"):
    ts_start = 0
    found_file = False
    if os.path.exists("./model/"):
        for file in os.listdir("./model/"):
            gs_ = list(map(int, file.split("_")[0].split("x")))
            ts_ = int(file.split("_")[1])
            if ts_ >= timestep_start \
                    and gs_[0] == grid_size[0] \
                    and gs_[1] == grid_size[1]:
                ts_start = ts_
                found_file = True
    if found_file:
        return f"/home/docker_user/model/{grid_size[0]}x{grid_size[1]}_{ts_start}_{model_type}"
    return False


if __name__ == "__main__":
    _curriculum_transitions = [  # number of steps to next stage, (grid_dims)
            (int(1e6), (4, 4)),
            (int(2e6), (6, 5)),
            (int(4e6), (8, 7)),
            (int(10e6), (10, 9)),
        ]

    tb_log_path = "/home/docker_user/logs/"
    env_id = "gym_environment/Snake-v0"

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ===============Edit below=================
    render_fps = 16
    grid_size = (10, 9)
    window_size = (800, 600)
    testing_episode_count = int(1)
    training_timesteps = max([i[0] for i in _curriculum_transitions])
    n_envs = 8

    load_pretrained = False
    model_type = "MPPO"
    ent_coef = 0.001  # sb default is 0
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
    )

    timestep_start = 0
    window_size = (
        window_size[0] // int(np.sqrt(n_envs)),
        window_size[1] // int(np.sqrt(n_envs)),
    )

    if not os.path.exists("/home/docker_user/model/"):
        os.makedirs("/home/docker_user/model/")

    env = make_vec_env(
        env_id,
        n_envs=n_envs,
        env_kwargs=dict(
            grid_size=grid_size,
        ),
        vec_env_cls=SubprocVecEnv,
    )

    if load_pretrained:
        file_name = get_model_filename(grid_size)
        if file_name:
            model = MaskablePPO.load(
                file_name,
                env,
                verbose=1,
                device=device,
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
        else:
            print("""
                  No pretrained model found,
                  please change option load_pretrained to False
                  and set the grid size to (4, 4) in order to start curriculum training""")
            exit()
    else:
        model = MaskablePPO(
            "CnnPolicy",
            env,
            verbose=1,
            device=device,
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
        model.learn(total_timesteps=training_timesteps,
                    reset_num_timesteps=False,
                    callback=CurriculumLearningCallback(
                        env_id=env_id,
                        device=device,
                        ent_coef=ent_coef,
                        tensorboard_log=tb_log_path,
                        learning_rate=learning_rate,
                        n_steps=n_steps,
                        batch_size=batch_size,
                        n_epochs=n_epochs,
                        gamma=gamma,
                        gae_lambda=gae_lambda,
                        clip_range=clip_range,
                        model_type=model_type,
                        stage=0,
                        curriculum_transitions=_curriculum_transitions,
                        )
                    )
        for stage in range(1, 4):
            file_name = f"/home/docker_user/model/{stage-1}"
            del model
            grid_size = _curriculum_transitions[stage][1]
            env = make_vec_env(
                env_id,
                n_envs=n_envs,
                env_kwargs=dict(
                    grid_size=grid_size,
                ),
                vec_env_cls=SubprocVecEnv,
            )
            model = MaskablePPO.load(
                file_name,
                env=env,
                verbose=1,
                device=device,
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
            model.learn(total_timesteps=training_timesteps,
                        callback=CurriculumLearningCallback(
                            env_id=env_id,
                            device=device,
                            ent_coef=ent_coef,
                            tensorboard_log=tb_log_path,
                            learning_rate=learning_rate,
                            n_steps=n_steps,
                            batch_size=batch_size,
                            n_epochs=n_epochs,
                            gamma=gamma,
                            gae_lambda=gae_lambda,
                            clip_range=clip_range,
                            model_type=model_type,
                            stage=stage,
                            curriculum_transitions=_curriculum_transitions,
                            )
                        )

        if training_timesteps + timestep_start > 0:
            model_filename = f"""/home/docker_user/model/\
                {grid_size[0]}x{grid_size[1]}_\
                {timestep_start + training_timesteps}_\
                {model_type}"""
            model.save(model_filename)

    if load_pretrained:
        cv2.startWindowThread()
        cv2.namedWindow("game")
        env = make_vec_env(
            env_id,
            n_envs=n_envs,
            env_kwargs=dict(
                grid_size=grid_size,
                render_window_size=window_size,
                training_mode=False
            ),
            vec_env_cls=SubprocVecEnv,
        )
        model.set_env(env)

        observations = model.get_env().reset()
        for _ in range(testing_episode_count):
            observations = env.reset()
            terminated = np.zeros(n_envs, dtype=bool)
            steps = np.zeros(n_envs, dtype=int)
            rewards = np.zeros(n_envs)

            while not np.all(terminated):
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
