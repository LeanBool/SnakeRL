import torch # type: ignore
import gymnasium # type: ignore
import gym_environment

if __name__ == '__main__':
    print("CUDA Available:", torch.cuda.is_available())
    env = gymnasium.make('gym_environment/Snake-v0')