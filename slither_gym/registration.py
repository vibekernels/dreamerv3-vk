from gymnasium.envs.registration import register

register(
    id="Slither-v0",
    entry_point="slither_gym.env.slither_env:SlitherEnv",
    max_episode_steps=4000,
)
