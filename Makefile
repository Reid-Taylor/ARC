# Makefile for ARC Encoder Flyte Deployment

# Configuration variables
REGISTRY ?= your-registry.com
PROJECT ?= arc-training
DOMAIN ?= development
TAG ?= latest
CONFIG ?= flyte_workflows/config.yaml
LOCAL_CONFIG ?= flyte_workflows/config-local.yaml

# Training parameters
EPOCHS ?= 10
BATCH_SIZE ?= 32
LEARNING_RATE ?= 1e-3
ALPHA ?= 0.85
DATASET_PATH ?= training

# Make deployment script executable
setup:
	@echo "Setting up Flyte deployment scripts..."
	chmod +x scripts/flyte_deploy.py
	chmod +x scripts/train_encoder.py

# Local testing
test-local:
	@echo "Running local test of ARC Encoder training..."
	python scripts/train_encoder.py \
		--epochs 2 \
		--batch-size 4 \
		--learning-rate $(LEARNING_RATE) \
		--alpha $(ALPHA) \
		--dataset-path $(DATASET_PATH) \
		--model-save-path ./models/test \
		--log-path ./logs/test

# View local results
test-local-results:
	tensorboard --logdir logs/test/arc_encoder
	
# Test Flyte workflow locally
test-flyte:
	@echo "Testing Flyte workflow locally..."
	python scripts/flyte_deploy.py test

# Build and push Docker image
build:
	@echo "Building and pushing Docker image..."
	python scripts/flyte_deploy.py build --registry $(REGISTRY) --tag $(TAG)

# Register workflow with Flyte cluster
register:
	@echo "Registering workflow with Flyte cluster..."
	python scripts/flyte_deploy.py register \
		--project $(PROJECT) \
		--domain $(DOMAIN) \
		--config $(CONFIG)

# Launch training workflow on Flyte
launch:
	@echo "Launching training workflow..."
	python scripts/flyte_deploy.py launch \
		--project $(PROJECT) \
		--domain $(DOMAIN) \
		--config $(CONFIG) \
		--epochs $(EPOCHS) \
		--batch-size $(BATCH_SIZE) \
		--learning-rate $(LEARNING_RATE) \
		--alpha $(ALPHA) \
		--dataset-path $(DATASET_PATH)

# Full deployment pipeline
deploy:
	@echo "Running full deployment pipeline..."
	python scripts/flyte_deploy.py deploy \
		--registry $(REGISTRY) \
		--tag $(TAG) \
		--project $(PROJECT) \
		--domain $(DOMAIN) \
		--config $(CONFIG) \
		--epochs $(EPOCHS) \
		--batch-size $(BATCH_SIZE) \
		--learning-rate $(LEARNING_RATE) \
		--alpha $(ALPHA) \
		--dataset-path $(DATASET_PATH)

# Install Flyte dependencies
install:
	@echo "Installing Flyte dependencies..."
	pip install -r requirements-flyte.txt

# Check Flyte cluster status
status:
	@echo "Checking Flyte cluster status..."
	flytectl get projects --config $(CONFIG)

# View running executions
executions:
	@echo "Viewing running executions..."
	flytectl get executions --project $(PROJECT) --domain $(DOMAIN) --config $(CONFIG)

# Clean up local artifacts
clean:
	@echo "Cleaning up local artifacts..."
	rm -rf ./models/test
	rm -rf ./logs/test
	rm -rf lightning_logs/
	docker image prune -f

# Help target
help:
	@echo "Available targets:"
	@echo "  setup       - Make scripts executable"
	@echo "  test-local  - Test training script locally (quick test)"
	@echo "  test-flyte  - Test Flyte workflow locally"
	@echo "  build       - Build and push Docker image"
	@echo "  register    - Register workflow with Flyte cluster"
	@echo "  launch      - Launch training workflow"
	@echo "  deploy      - Full deployment (build + register + launch)"
	@echo "  install     - Install Flyte dependencies"
	@echo "  status      - Check Flyte cluster status"
	@echo "  executions  - View running executions"
	@echo "  clean       - Clean up local artifacts"
	@echo ""
	@echo "Configuration variables (can be overridden):"
	@echo "  REGISTRY=$(REGISTRY)"
	@echo "  PROJECT=$(PROJECT)"
	@echo "  DOMAIN=$(DOMAIN)"
	@echo "  TAG=$(TAG)"
	@echo "  EPOCHS=$(EPOCHS)"
	@echo "  BATCH_SIZE=$(BATCH_SIZE)"
	@echo "  LEARNING_RATE=$(LEARNING_RATE)"
	@echo "  ALPHA=$(ALPHA)"
	@echo ""
	@echo "Example usage:"
	@echo "  make deploy REGISTRY=myregistry.com EPOCHS=20 BATCH_SIZE=64"

.PHONY: setup test-local test-flyte build register launch deploy install status executions clean help