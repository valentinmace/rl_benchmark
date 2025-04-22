#!/bin/bash
# Run evaluation for a specific trained model

# Define colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[0;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Check if required arguments are provided
if [ $# -lt 3 ]; then
    echo -e "${YELLOW}Usage: $0 <environment> <algorithm> <timestamp>${NC}"
    echo -e "Example: $0 CartPole-v1 ppo 20230415_120530"
    exit 1
fi

# Get arguments
env=$1
algo=$2
timestamp=$3

# Get absolute path of workspace
WORKSPACE_DIR=$(pwd)

# Define output directory and model path using absolute paths
output_dir="${WORKSPACE_DIR}/outputs/${env}/${algo}/${timestamp}"
model_file="${output_dir}/model/model.zip"
model_path="${output_dir}/model/model"  # Path without .zip for SB3
config_file="${WORKSPACE_DIR}/configs/${env}/${algo}.yaml"

# Check if the model file exists
if [ -f "${model_file}" ]; then
    echo -e "${GREEN}Found model at ${model_file}${NC}"

    # Ensure the file has proper permissions
    echo -e "${BLUE}Ensuring proper permissions...${NC}"
    chmod 644 "${model_file}"

    echo -e "${GREEN}Using absolute path: ${model_path}${NC}"
else
    echo -e "${YELLOW}Error: Model file not found at ${model_file}${NC}"
    echo -e "Please check if the environment, algorithm and timestamp are correct."
    exit 1
fi

echo -e "${BLUE}Evaluating ${algo} on ${env} (Run ID: ${timestamp})...${NC}"
echo -e "${GREEN}Using config: ${config_file}${NC}"

# Run evaluation with corrected Hydra syntax
echo -e "${GREEN}Running evaluation...${NC}"
cd ${WORKSPACE_DIR}  # Ensure we're in the workspace directory
python -m training.evaluate --config-path="${WORKSPACE_DIR}/configs/${env}" --config-name=${algo} "hydra.run.dir=${output_dir}/hydra" "+save_path=${model_path}"

echo -e "${BLUE}Evaluation completed!${NC}"
