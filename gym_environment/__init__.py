from gymnasium.envs.registration import register # type: ignore

register(
    id="gym_environment/Snake-v0",
    entry_point="gym_environment.envs:SnakeEnv",
)