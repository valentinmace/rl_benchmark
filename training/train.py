import logging
import os
import time

import hydra
from omegaconf import DictConfig, OmegaConf
from stable_baselines3.common.logger import configure

from agents.agent_factory import create_agent
from envs.make_env import make_env

logger = logging.getLogger(__name__)


@hydra.main(config_path="../configs", config_name=None, version_base=None)
def train_agent(cfg: DictConfig) -> None:
    """
    Train an RL agent using the specified configuration.

    Parameters:
    - cfg (DictConfig): Configuration object containing training parameters.
    """
    # Print configuration
    print("=== Training Configuration ===")
    print(f"Environment: {cfg.env_id}")
    print(f"Algorithm: {cfg.agent_type}")
    print(f"Total timesteps: {cfg.total_timesteps}")
    print(f"Save path: {cfg.save_path}")
    print("===========================")

    # Initialize environment
    print(f"Initializing environment {cfg.env_id}...")
    env = make_env(cfg.env_id, cfg.seed)
    print("Environment initialized.")

    # Convert DictConfig to regular dict
    agent_params = OmegaConf.to_container(cfg, resolve=True)

    # Remove non-agent parameters
    for param in [
        "env_id",
        "seed",
        "total_timesteps",
        "log_interval",
        "save_path",
        "agent_type",
    ]:
        if param in agent_params:
            del agent_params[param]

    # Add verbosity to see training progress
    if "verbose" not in agent_params:
        agent_params["verbose"] = 1

    # Create agent with all parameters - agent_factory will filter the relevant ones
    print(f"Creating {cfg.agent_type} agent...")
    agent = create_agent(cfg.agent_type, env, **agent_params)
    print("Agent created successfully.")

    # Get run directory (parent of model directory)
    run_dir = os.path.dirname(os.path.dirname(cfg.save_path))

    # Create necessary directories
    os.makedirs(os.path.dirname(cfg.save_path), exist_ok=True)

    # Set up logging in the logs directory
    log_dir = os.path.join(run_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    print(f"Logs will be saved to: {log_dir}")

    # Configure logger to save both CSV and TensorBoard logs
    custom_logger = configure(log_dir, ["csv", "tensorboard"])
    agent.set_logger(custom_logger)

    # Train agent
    print(f"Starting training for {cfg.total_timesteps} timesteps...")
    start_time = time.time()
    agent.learn(total_timesteps=cfg.total_timesteps, log_interval=cfg.log_interval)
    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.2f} seconds.")

    # Save model
    print(f"Saving model to {cfg.save_path}...")
    agent.save(cfg.save_path)
    print("Model saved successfully.")

    # Create a README file with run information
    readme_path = os.path.join(run_dir, "README.md")
    with open(readme_path, "w") as f:
        f.write(f"# Training Run: {os.path.basename(run_dir)}\n\n")
        f.write(f"- **Environment:** {cfg.env_id}\n")
        f.write(f"- **Algorithm:** {cfg.agent_type}\n")
        f.write(f"- **Timesteps:** {cfg.total_timesteps}\n")
        f.write(f"- **Training Duration:** {training_time:.2f} seconds\n\n")
        f.write("## Directories\n\n")
        f.write("- `model/`: Contains the trained model\n")
        f.write("- `logs/`: Contains CSV and TensorBoard logs\n")
        f.write("- `hydra/`: Contains Hydra configuration\n")
        f.write("- `eval_results/`: If present, contains evaluation results\n")

    print(f"Created README at: {readme_path}")


if __name__ == "__main__":
    train_agent()
