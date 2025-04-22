import datetime
import json
import os
from typing import List

import hydra
import numpy as np
from omegaconf import DictConfig
from tqdm import tqdm

from agents.agent_factory import load_agent
from envs.make_env import make_env


@hydra.main(config_path="../configs", config_name=None, version_base=None)
def evaluate_agent(cfg: DictConfig) -> None:
    """
    Evaluate a trained RL agent using the specified configuration.

    Parameters:
    - cfg (DictConfig): Configuration object containing evaluation parameters.
    """
    # Initialize environment
    env = make_env(cfg.env_id, cfg.seed)

    # Load agent using the agent_factory
    agent_type = cfg.agent_type
    save_path = cfg.save_path

    # Load the agent using the factory function
    agent = load_agent(agent_type, save_path)

    # Evaluation parameters
    n_eval_steps = 1000

    # Evaluate agent
    obs, info = env.reset()
    episode_rewards = []
    episode_lengths = []
    episode_length = 0
    episode_reward = 0.0

    print(f"Evaluating {agent_type} agent on {cfg.env_id} for {n_eval_steps} steps...")

    # Run evaluation
    for _ in tqdm(range(n_eval_steps), desc="Evaluation progress", ncols=100):
        action, _states = agent.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)
        episode_reward += reward
        episode_length += 1

        if done or truncated:
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            episode_reward = 0.0
            episode_length = 0
            obs, info = env.reset()

        env.render()

    # Calculate statistics
    avg_reward = sum(episode_rewards) / len(episode_rewards) if episode_rewards else 0
    avg_length = sum(episode_lengths) / len(episode_lengths) if episode_lengths else 0

    # Print evaluation results
    print("\nEvaluation Results:")
    print(f"  - Number of episodes completed: {len(episode_rewards)}")
    print(f"  - Average episode reward: {avg_reward:.2f}")
    print(f"  - Average episode length: {avg_length:.2f}")

    if episode_rewards:
        print(f"  - Min episode reward: {min(episode_rewards):.2f}")
        print(f"  - Max episode reward: {max(episode_rewards):.2f}")

    # Save results to file
    save_results(cfg, episode_rewards, episode_lengths)

    print("\nEvaluation completed!")


def save_results(
    cfg: DictConfig, episode_rewards: List[float], episode_lengths: List[int]
) -> None:
    """
    Save evaluation results to a JSON file in the model directory.

    Parameters:
    - cfg (DictConfig): Configuration with paths and parameters
    - episode_rewards (List[float]): List of rewards for each episode
    - episode_lengths (List[int]): List of episode lengths
    """
    # Derive output directory from save_path
    # Expected format: outputs/ENV/ALGO/TIMESTAMP/model/model
    save_path = cfg.save_path

    # Extract the base directory (remove /model/model from the end)
    base_dir = os.path.dirname(os.path.dirname(save_path))

    # Create eval_results directory if it doesn't exist
    eval_dir = os.path.join(base_dir, "eval_results")
    os.makedirs(eval_dir, exist_ok=True)

    # Generate a timestamp for the eval file
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # Create results dictionary
    results = {
        "env_id": cfg.env_id,
        "agent_type": cfg.agent_type,
        "eval_timestamp": timestamp,
        "model_path": save_path,
        "num_episodes": len(episode_rewards),
        "episode_rewards": episode_rewards,
        "episode_lengths": episode_lengths,
        "statistics": {
            "mean_reward": float(np.mean(episode_rewards)) if episode_rewards else 0,
            "median_reward": (
                float(np.median(episode_rewards)) if episode_rewards else 0
            ),
            "std_reward": float(np.std(episode_rewards)) if episode_rewards else 0,
            "min_reward": float(np.min(episode_rewards)) if episode_rewards else 0,
            "max_reward": float(np.max(episode_rewards)) if episode_rewards else 0,
            "mean_length": float(np.mean(episode_lengths)) if episode_lengths else 0,
        },
    }

    # Save to JSON file
    eval_file = os.path.join(eval_dir, f"eval_{timestamp}.json")
    with open(eval_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Evaluation results saved to: {eval_file}")


if __name__ == "__main__":
    evaluate_agent()
