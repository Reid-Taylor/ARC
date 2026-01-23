GITHUB_USERNAME ?= Reid-Taylor
REGISTRY ?= ghcr.io/$(GITHUB_USERNAME)
PROJECT ?= arc-training
DOMAIN ?= development
TAG ?= latest
GCP_ZONE ?= us-central1-c

# Training parameters
EPOCHS ?= 250
BATCH_SIZE ?= 512
LEARNING_RATE ?= 1e-3
ALPHA ?= 0.75
DATASET_PATH ?= training
MODEL ?= encoder

gcp-create:
	@echo "Creating GCP instance..."
	gcloud compute instances create arc-training-vm \
		--project amplified-hull-484821-b5 \
		--zone=$(GCP_ZONE) \
		--machine-type n1-standard-4 \
		--boot-disk-size 100GB \
		--maintenance-policy TERMINATE \
		--restart-on-failure \

gcp-start:
	gcloud compute instances start arc-training-vm --zone=$(GCP_ZONE)

gcp-stop:
	gcloud compute instances stop arc-training-vm --zone=$(GCP_ZONE)

gcp-ssh:
	gcloud compute ssh arc-training-vm --zone=$(GCP_ZONE)

gcp-deploy:
	@echo "Deploying code to GCP instance..."
	gcloud compute scp --recurse . arc-training-vm:~/ARC --zone=$(GCP_ZONE)
	gcloud compute ssh arc-training-vm --zone=$(GCP_ZONE) --command="cd ARC && ./scripts/setup_gcp_instance.sh"

gcp-train:
	gcloud compute ssh arc-training-vm --zone=$(GCP_ZONE) --command="cd ARC && source .venv/bin/activate && python scripts/train_encoder.py --epochs 100 --batch-size 64"

train-encoder:
	@echo "Running full pipeline of ARC Encoder training..."
	python scripts/train_encoder.py \
		--config train_config \
		--epochs $(EPOCHS) \
		--batch-size $(BATCH_SIZE) \
		--learning-rate $(LEARNING_RATE) \
		--alpha $(ALPHA) \
		--dataset-path $(DATASET_PATH) \
		--model-save-path ./models/test \
		--log-path ./logs/test

local-test-encoder:
	@echo "Running local test of ARC Encoder training..."
	python scripts/train_encoder.py \
		--config test_config \
		--epochs 2 \
		--batch-size $(BATCH_SIZE) \
		--learning-rate $(LEARNING_RATE) \
		--alpha $(ALPHA) \
		--dataset-path $(DATASET_PATH) \
		--model-save-path ./models/test \
		--log-path ./logs/test

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

local-view-training:
	tensorboard --logdir logs/test/arc_$(MODEL)

clean:
	@echo "Cleaning up local artifacts..."
	rm -rf ./models/test
	rm -rf ./logs/test
	rm -rf lightning_logs/
	docker image prune -f

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