SHELL := /bin/bash

# Variables
# Use the shell command pwd to get the current directory explicitly
WORK_DIR := $(shell pwd)
DOCKER_IMAGE_NAME = rl_benchmark:latest
DOCKER_RUN_FLAGS = --rm -it

# Port for TensorBoard (internal:external)
TENSORBOARD_PORT_INTERNAL = 6006
TENSORBOARD_PORT_EXTERNAL = 8754

.PHONY: clean
clean:
	find . -type f -name "*.DS_Store" -ls -delete
	find . | grep -E "(__pycache__|\.pyc|\.pyo)" | xargs rm -rfv
	find . | grep -E ".ipynb_checkpoints" | xargs rm -rfv

.PHONY: build
build:
	docker build --no-cache -t $(DOCKER_IMAGE_NAME) .

.PHONY: shell
shell:
	@echo "Mounting directory: $(WORK_DIR)"
	docker run $(DOCKER_RUN_FLAGS) -v $(WORK_DIR):/app -p $(TENSORBOARD_PORT_EXTERNAL):$(TENSORBOARD_PORT_INTERNAL) $(DOCKER_IMAGE_NAME)

# Helper command to display available commands
.PHONY: help
help:
	@echo "Available commands:"
	@echo "  make build            - Build the Docker image"
	@echo "  make shell            - Run an interactive shell in the Docker container"
	@echo "  make clean            - Clean Python cache files"
	@echo "  make help             - Show this help message"
