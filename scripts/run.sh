#!/bin/bash
# Run experiments for different environments and algorithms

# Function to create directories for a specific environment and algorithm
create_directories() {
    env=$1
    algo=$2
    timestamp=$3

    # Create unified output directory structure
    mkdir -p outputs/${env}/${algo}/${timestamp}/model
    mkdir -p outputs/${env}/${algo}/${timestamp}/logs
}

# Define colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[0;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Get absolute path of workspace
WORKSPACE_DIR=$(pwd)

# Function to run an experiment
run_experiment() {
    env=$1
    algo=$2

    # Generate timestamp for this run
    timestamp=$(date +"%Y%m%d_%H%M%S")

    # Create necessary directories
    create_directories "${env}" "${algo}" "${timestamp}"

    # Define dynamically the output paths with absolute paths
    output_dir="${WORKSPACE_DIR}/outputs/${env}/${algo}/${timestamp}"
    save_path="${output_dir}/model/model"
    config_file="${WORKSPACE_DIR}/configs/${env}/${algo}.yaml"

    echo -e "${BLUE}Running ${algo} on ${env} (Run ID: ${timestamp})...${NC}"

    # Train the agent with simple command-line overrides
    echo -e "${GREEN}Training...${NC}"
    cd ${WORKSPACE_DIR}  # Ensure we're in the workspace directory

    # Create a simple config that just references the env and algo parameters
    echo -e "${GREEN}Using config: ${config_file}${NC}"
    python -m training.train --config-path="${WORKSPACE_DIR}/configs/${env}" --config-name=${algo} "hydra.run.dir=${output_dir}/hydra" "+save_path=${save_path}"

    # Make sure model file has correct permissions if it exists
    model_file="${save_path}.zip"
    if [ -f "${model_file}" ]; then
        chmod 644 "${model_file}"
        echo -e "${GREEN}Model saved successfully at ${model_file}${NC}"
    else
        echo -e "${YELLOW}Warning: Model file not found at ${model_file}${NC}"
        echo -e "${YELLOW}Evaluation may fail if no model is available.${NC}"
    fi

    # Evaluate the agent
    echo -e "${GREEN}Evaluating...${NC}"
    python -m training.evaluate --config-path="${WORKSPACE_DIR}/configs/${env}" --config-name=${algo} "hydra.run.dir=${output_dir}/hydra" "+save_path=${save_path}"

    echo -e "${BLUE}Finished ${algo} on ${env}${NC}"
    echo "------------------------------------"
}

# Check if specific environment and algorithm are provided
if [ $# -eq 2 ]; then
    run_experiment $1 $2
    exit 0
fi

# Run all combinations if no specific arguments
echo -e "${BLUE}Running all experiments...${NC}"

# CartPole experiments
run_experiment "CartPole-v1" "ppo"
run_experiment "CartPole-v1" "dqn"
run_experiment "CartPole-v1" "a2c"

# LunarLander experiments
run_experiment "LunarLander-v3" "ppo"
run_experiment "LunarLander-v3" "dqn"
run_experiment "LunarLander-v3" "a2c"

# MountainCar experiments
run_experiment "MountainCar-v0" "ppo"
run_experiment "MountainCar-v0" "dqn"
run_experiment "MountainCar-v0" "a2c"

echo -e "${BLUE}All experiments completed successfully!${NC}"
echo "You can generate a report using: python -m scripts.generate_report"
