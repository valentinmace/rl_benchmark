import os
import tempfile
from typing import Any, Dict

import gymnasium as gym
import pytest
from stable_baselines3 import A2C, DQN, PPO

from agents.agent_factory import AGENT_REGISTRY, create_agent, load_agent


def test_create_ppo_agent():
    """Test creating a PPO agent."""
    env = gym.make("CartPole-v1")
    agent = create_agent("PPO", env)
    assert isinstance(agent, PPO), "Agent should be a PPO instance"


def test_create_dqn_agent():
    """Test creating a DQN agent."""
    env = gym.make("CartPole-v1")
    agent = create_agent("DQN", env)
    assert isinstance(agent, DQN), "Agent should be a DQN instance"


def test_create_a2c_agent():
    """Test creating an A2C agent."""
    env = gym.make("CartPole-v1")
    agent = create_agent("A2C", env)
    assert isinstance(agent, A2C), "Agent should be an A2C instance"


def test_invalid_agent_type():
    """Test that an invalid agent type raises a ValueError."""
    env = gym.make("CartPole-v1")
    with pytest.raises(ValueError):
        create_agent("INVALID_TYPE", env)


def test_agent_registry_contains_expected_agents():
    """Test that the agent registry contains the expected agents."""
    expected_agents = {"PPO", "DQN", "A2C"}
    registered_agents = set(AGENT_REGISTRY.keys())
    assert expected_agents.issubset(
        registered_agents
    ), "Agent registry should contain all expected agents"


def test_agent_parameter_filtering():
    """Test that agent parameters are correctly filtered."""
    env = gym.make("CartPole-v1")
    # Mix of valid and invalid parameters for PPO
    params: Dict[str, Any] = {
        "learning_rate": 0.001,
        "gamma": 0.99,
        "n_steps": 2048,
        "invalid_param": "value",  # This should be filtered out
        "buffer_size": 10000,  # This is for DQN, should be filtered out for PPO
    }
    agent = create_agent("PPO", env, **params)
    assert isinstance(
        agent, PPO
    ), "Agent should be a PPO instance despite invalid parameters"


def test_agent_predict():
    """Test that the agent can make predictions."""
    env = gym.make("CartPole-v1")
    agent = create_agent("PPO", env)

    obs, _ = env.reset()
    action, _states = agent.predict(obs, deterministic=True)

    assert action in [0, 1], "Action should be valid for CartPole (0 or 1)"


def test_load_agent():
    """Test that the agent can be saved and loaded."""
    # Create and save an agent
    env = gym.make("CartPole-v1")
    agent = create_agent("PPO", env)

    # Use a temporary directory for saving
    with tempfile.TemporaryDirectory() as tmp_dir:
        save_path = os.path.join(tmp_dir, "test_agent")
        agent.save(save_path)

        # Test loading the agent
        loaded_agent = load_agent("PPO", save_path)
        assert isinstance(loaded_agent, PPO), "Loaded agent should be a PPO instance"

        # Test that loaded agent can make predictions
        obs, _ = env.reset()
        action, _states = loaded_agent.predict(obs, deterministic=True)
        assert action in [
            0,
            1,
        ], "Action from loaded agent should be valid for CartPole (0 or 1)"


def test_load_agent_invalid_type():
    """Test that loading an agent with an invalid type raises a ValueError."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        save_path = os.path.join(tmp_dir, "test_agent")
        with pytest.raises(ValueError):
            load_agent("INVALID_TYPE", save_path)


if __name__ == "__main__":
    pytest.main()
