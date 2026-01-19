GITHUB_USERNAME ?= Reid-Taylor  # Replace with your GitHub username
REGISTRY ?= ghcr.io/$(GITHUB_USERNAME)  # GitHub Container Registry (free)
PROJECT ?= arc-training
DOMAIN ?= development
TAG ?= latest

# Training parameters
EPOCHS ?= 1000
BATCH_SIZE ?= 32
LEARNING_RATE ?= 1e-3
ALPHA ?= 0.85
DATASET_PATH ?= training
MODEL ?= encoder

# Local testing
local-test-encoder:
	@echo "Running local test of ARC Encoder training..."
	python scripts/train_encoder.py \
		--epochs 2 \
		--batch-size 4 \
		--learning-rate $(LEARNING_RATE) \
		--alpha $(ALPHA) \
		--dataset-path $(DATASET_PATH) \
		--model-save-path ./models/test \
		--log-path ./logs/test

# Local testing
local-test-transformer:
	@echo "Running local test of ARC Encoder training..."
	python scripts/train_transformer.py \
		--epochs 2 \
		--batch-size 4 \
		--learning-rate $(LEARNING_RATE) \
		--alpha $(ALPHA) \
		--dataset-path $(DATASET_PATH) \
		--model-save-path ./models/test \
		--log-path ./logs/test

# View local results
local-view-training:
	tensorboard --logdir logs/test/arc_$(MODEL)

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
	@echo "  test-local  - Test training script locally (quick test)"
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

.PHONY: test-local local-view-training clean help