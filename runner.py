import torch # type: ignore
import gymnasium # type: ignore
import gym_environment

if __name__ == '__main__':
    print("CUDA Available:", torch.cuda.is_available())
    env = gymnasium.make('gym_environment/Snake-v0', render_mode="human")
    observation, info = env.reset()
    for _ in range(500):
        observation, reward, terminated, _, info = env.step(1)

        if terminated:
            observation, info = env.reset()

