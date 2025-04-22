from typing import Optional

import gymnasium as gym


def make_env(
    env_id: str, seed: Optional[int] = None, render_mode: Optional[str] = None
) -> gym.Env:
    """
    Create and configure a gymnasium environment.

    Parameters:
    - env_id (str): The ID of the gym environment to create.
    - seed (int, optional): The seed for random number generation.
    - render_mode (str, optional): The render mode to use.
    Default is None for headless environments.

    Returns:
    - gym.Env: The configured gym environment.
    """

    # Create the environment with the specified render mode
    env = gym.make(env_id, render_mode=render_mode)

    # Initialize the environment with the seed
    if seed is not None:
        env.reset(seed=seed)

    # Additional wrappers or configurations can be added here
    return env


# Example usage
if __name__ == "__main__":
    env = make_env("CartPole-v1", seed=42, render_mode="human")
    obs, info = env.reset()
    print("Initial observation:", obs)
