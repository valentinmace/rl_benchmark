# RL Benchmark Suite

![CI Status](https://github.com/valentinmace/rl_benchmark/actions/workflows/tests.yml/badge.svg)

## Overview
This repository provides a suite for benchmarking reinforcement learning (RL) agents using stable-baselines3 on classic RL environments. The goal is to demonstrate clean, modular code with best practices in CI/CD, unit testing, and reporting.

## Project Structure
- `agents/`: Wrappers around SB3 algorithms.
- `envs/`: Setup and configuration of environments.
- `training/`: Scripts for training and evaluation.
- `configs/`: YAML configurations for hyperparameters and environments.
- `tests/`: Pytest for critical components.
- `scripts/`: Scripts to run experiments and generate reports.
- `reports/`: Logs, plots, and stats.
- `.github/workflows/`: CI with GitHub Actions.

## Getting Started

### Prerequisites
- Python 3.10
- Docker (optional for containerized setup)

### Installation
1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd rl_benchmark
   ```
2. Install the required Python packages:
   ```bash
   pip install -r requirements.txt
   ```
3. Set up pre-commit hooks:
   ```bash
   pre-commit install
   ```

## Usage

### Training
To train an agent, run:
```bash
python training/train.py hydra.run.dir=./outputs +config_path=../configs/CartPole-v1/ppo.yaml
```

### Evaluation
To evaluate a trained agent, run:
```bash
python training/evaluate.py hydra.run.dir=./outputs +config_path=../configs/CartPole-v1/ppo.yaml
```

## Contributing

### Code Style
This project uses `black` and `flake8` for code formatting and linting. Ensure your code is formatted and linted before committing.

### Pre-commit Hooks
We use pre-commit hooks to automate code quality checks. Make sure to set up pre-commit hooks as described in the installation section.
