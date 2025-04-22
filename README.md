# RL Benchmark Suite

![CI Status](https://github.com/valentinmace/rl_benchmark/actions/workflows/tests.yml/badge.svg)

## Overview

This repository serves as a personal demonstration of software engineering good practices in the context of reinforcement learning (RL). Rather than focusing on novel algorithms, this project emphasizes:

- Clean, type-annotated code
- Continuous integration and automated testing
- Configuration management with Hydra
- Standardized code formatting and linting
- Reproducible environments with Docker
- Proper dependency management

## Project Structure

- `agents/`: Algorithm implementations and wrappers for Stable Baselines 3.
- `configs/`: YAML configuration files using Hydra for hyperparameter management.
- `envs/`: Environment setup and configuration utilities.
- `training/`: Core training and evaluation logic.
- `scripts/`: Automation scripts for running experiments.
- `tests/`: Unit and integration tests.
- `outputs/`: Generated models, logs, and evaluation results (created at runtime).
- `.github/workflows/`: CI pipeline definitions.

## Getting Started

### Prerequisites

- Python 3.10+
- Docker (recommended for containerized execution)

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/valentinmace/rl_benchmark.git
   cd rl_benchmark
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up pre-commit hooks for code quality:
   ```bash
   pre-commit install
   ```

### Docker Setup (Recommended)

For consistent environments across systems:

```bash
# Build the Docker image
make build

# Run the container with shell access
make shell
```

## Usage

### Running Experiments

The project includes a convenient script to run training and evaluation for multiple environments and algorithms:

```bash
# Run a specific environment-algorithm combination
./scripts/run.sh CartPole-v1 ppo

# Or run all predefined combinations
./scripts/run.sh
```

### Evaluating Trained Models

To evaluate a previously trained model:

```bash
# Format: ./scripts/run_eval.sh <environment> <algorithm> <timestamp>
./scripts/run_eval.sh CartPole-v1 ppo 20230415_120530
```

### Output Structure

All experiment outputs are organized in a consistent directory structure:

```
outputs/
└── <environment>/
    └── <algorithm>/
        └── <timestamp>/
            ├── model/         # Saved model files
            ├── logs/          # Training logs (CSV and TensorBoard)
            ├── hydra/         # Hydra configuration logs
            └── eval_results/  # Evaluation metrics and data
```

## Development Practices

### Code Style

This project strictly adheres to:
- [PEP 8](https://pep8.org/) style guide
- Type annotations as per [PEP 484](https://peps.python.org/pep-0484/)
- Code formatting with [Black](https://black.readthedocs.io/)
- Linting with [Flake8](https://flake8.pycqa.org/)

### Testing

Tests are written using pytest and automatically run through GitHub Actions on every push and pull request to the main branch.

To run tests locally:

```bash
pytest
```

### Continuous Integration

The CI pipeline runs the following checks:
- Code linting with flake8
- Code formatting with black
- Unit and integration tests with pytest

## Supported Environments and Algorithms

### Environments
- CartPole-v1
- LunarLander-v3
- MountainCar-v0

### Algorithms
- PPO (Proximal Policy Optimization)
- DQN (Deep Q-Network)
- A2C (Advantage Actor-Critic)
