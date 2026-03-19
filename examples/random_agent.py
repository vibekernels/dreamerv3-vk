"""Smoke test: run a random agent in the Slither environment."""

import slither_gym  # noqa: F401 – registers the env
import gymnasium as gym


def main():
    env = gym.make("Slither-v0", render_mode="rgb_array")
    obs, info = env.reset(seed=42)
    print(f"Observation shape: {obs.shape}, dtype: {obs.dtype}")
    print(f"Action space: {env.action_space}")

    total_reward = 0.0
    for step in range(1000):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        if step % 100 == 0:
            print(f"Step {step}: reward={reward:.3f}, length={info['length']}, "
                  f"score={info['score']:.1f}, alive_npcs={info['alive_npcs']}")

        if terminated or truncated:
            print(f"Episode ended at step {step}: "
                  f"terminated={terminated}, truncated={truncated}")
            print(f"Total reward: {total_reward:.3f}")
            obs, info = env.reset()
            total_reward = 0.0

    env.close()
    print("Done!")


if __name__ == "__main__":
    main()
