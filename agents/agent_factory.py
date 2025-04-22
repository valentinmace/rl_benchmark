from typing import Any, Dict, Type, Union

import gymnasium as gym
from stable_baselines3 import A2C, DQN, PPO

# Define a registry of agent types and their respective classes
AGENT_REGISTRY: Dict[str, Type] = {
    "PPO": PPO,
    "DQN": DQN,
    "A2C": A2C,
}

# Define parameter groups for each agent type
AGENT_PARAMS: Dict[str, Dict[str, Any]] = {
    "PPO": {
        "required": ["n_steps", "batch_size"],
        "optional": ["clip_range", "ent_coef", "vf_coef"],
    },
    "DQN": {
        "required": [],
        "optional": [
            "buffer_size",
            "learning_starts",
            "batch_size",
            "tau",
            "target_update_interval",
        ],
    },
    "A2C": {
        "required": ["n_steps"],
        "optional": ["gae_lambda", "ent_coef", "vf_coef"],
    },
}

# Common parameters for all agents
COMMON_PARAMS = ["learning_rate", "gamma", "verbose"]


def create_agent(agent_type: str, env: gym.Env, **kwargs: Any) -> Union[PPO, DQN, A2C]:
    """
    Create and configure a stable-baselines3 agent with type-specific hyperparameters.

    Parameters:
    - agent_type (str): The type of agent to create (must be in AGENT_REGISTRY).
    - env (gym.Env): The environment in which the agent will be trained.
    - kwargs (Any): Additional hyperparameters for the agent.

    Returns:
    - Union[PPO, DQN, A2C]: The configured RL agent.

    Raises:
    - ValueError: If an unsupported agent type is provided.
    """
    if agent_type not in AGENT_REGISTRY:
        supported_types = list(AGENT_REGISTRY.keys())
        raise ValueError(
            f"Unsupported agent type: {agent_type}. Supported types: {supported_types}"
        )

    agent_class = AGENT_REGISTRY[agent_type]

    # Filter parameters based on what's relevant for this agent type
    filtered_params = {}

    # Add common parameters
    for param in COMMON_PARAMS:
        if param in kwargs and kwargs[param] is not None:
            filtered_params[param] = kwargs[param]

    # Add agent-specific parameters
    for param in (
        AGENT_PARAMS[agent_type]["required"] + AGENT_PARAMS[agent_type]["optional"]
    ):
        if param in kwargs and kwargs[param] is not None:
            filtered_params[param] = kwargs[param]

    # Create and return the agent
    return agent_class("MlpPolicy", env, **filtered_params)


def load_agent(agent_type: str, save_path: str) -> Union[PPO, DQN, A2C]:
    """
    Load a trained agent from the specified path.

    Parameters:
    - agent_type (str): The type of agent to load (must be in AGENT_REGISTRY).
    - save_path (str): Path to the saved model.

    Returns:
    - Union[PPO, DQN, A2C]: The loaded RL agent.

    Raises:
    - ValueError: If an unsupported agent type is provided.
    """
    if agent_type not in AGENT_REGISTRY:
        supported_types = list(AGENT_REGISTRY.keys())
        raise ValueError(
            f"Unsupported agent type: {agent_type}. Supported types: {supported_types}"
        )

    agent_class = AGENT_REGISTRY[agent_type]
    return agent_class.load(save_path)


# Example usage
if __name__ == "__main__":
    from envs.make_env import make_env

    env = make_env("CartPole-v1", seed=2303)
    agent = create_agent("PPO", env, verbose=1)
    print("Agent created:", agent)
