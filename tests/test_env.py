import gymnasium as gym
import numpy as np
import pytest

from envs.make_env import make_env


def test_make_env_creates_valid_env():
    """Test that make_env creates a valid gymnasium environment."""
    env = make_env("CartPole-v1")
    assert isinstance(env, gym.Env), "Created environment should be a gym.Env instance"


def test_env_reset_returns_valid_observation():
    """Test that the environment reset returns a valid observation."""
    env = make_env("CartPole-v1")
    obs, info = env.reset()
    assert isinstance(obs, np.ndarray), "Observation should be a numpy array"
    assert obs.shape == (4,), "CartPole observation should have shape (4,)"


def test_env_step_returns_valid_values():
    """Test that the environment step returns valid values."""
    env = make_env("CartPole-v1")
    obs, _ = env.reset()
    action = env.action_space.sample()
    next_obs, reward, done, truncated, info = env.step(action)

    assert isinstance(next_obs, np.ndarray), "Next observation should be a numpy array"
    assert next_obs.shape == (4,), "CartPole next observation should have shape (4,)"
    assert isinstance(reward, float), "Reward should be a float"
    assert isinstance(done, bool), "Done should be a boolean"
    assert isinstance(truncated, bool), "Truncated should be a boolean"
    assert isinstance(info, dict), "Info should be a dictionary"


def test_seed_reproducibility():
    """Test that setting a seed gives reproducible results."""
    env1 = make_env("CartPole-v1", seed=42)
    obs1, _ = env1.reset()

    env2 = make_env("CartPole-v1", seed=42)
    obs2, _ = env2.reset()

    np.testing.assert_array_equal(
        obs1, obs2, "Same seed should give same initial observation"
    )


def test_different_seeds_give_different_results():
    """Test that different seeds give different results."""
    env1 = make_env("CartPole-v1", seed=42)
    obs1, _ = env1.reset()

    env2 = make_env("CartPole-v1", seed=43)
    obs2, _ = env2.reset()

    with pytest.raises(AssertionError):
        np.testing.assert_array_equal(
            obs1, obs2, "Different seeds should give different initial observations"
        )


if __name__ == "__main__":
    pytest.main()
