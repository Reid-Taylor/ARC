GCP_ZONE ?= us-east4-a
VM_INSTANCE_NAME ?= arc-training-v2
MACHINE_TYPE ?= e2-highmem-4
DISK_SIZE ?= 100GB

# Training parameters
DATASET_PATH ?= training
MODEL ?= encoder

gcp-create:
	@echo "Creating GCP instance..."
	gcloud compute instances create $(VM_INSTANCE_NAME) \
		--project amplified-hull-484821-b5 \
		--zone=$(GCP_ZONE) \
		--machine-type $(MACHINE_TYPE) \
		--boot-disk-size $(DISK_SIZE) \
		--maintenance-policy TERMINATE \
		--restart-on-failure \

gcp-start:
	gcloud compute instances start $(VM_INSTANCE_NAME) --zone=$(GCP_ZONE)

gcp-stop:
	gcloud compute instances stop $(VM_INSTANCE_NAME) --zone=$(GCP_ZONE)

gcp-ssh:
	gcloud compute ssh $(VM_INSTANCE_NAME) --zone=$(GCP_ZONE)

train-encoder:
	@echo "Running full pipeline of ARC Encoder training..."
	python scripts/train_encoder.py \
		--config train_config \
		--dataset-path $(DATASET_PATH) \
		--model-save-path ./models/test \
		--log-path ./logs/test

local-test-encoder:
	@echo "Running local test of ARC Encoder training..."
	python scripts/train_encoder.py \
		--config test_config \
		--dataset-path $(DATASET_PATH) \
		--model-save-path ./models/test \
		--log-path ./logs/test

local-test-transformer:
	@echo "Running local test of ARC Encoder training..."
	python scripts/train_transformer.py \
		--dataset-path $(DATASET_PATH) \
		--model-save-path ./models/test \
		--log-path ./logs/test

view-training:
	tensorboard --logdir logs/test/arc_$(MODEL)

clean:
	@echo "Cleaning up local artifacts..."
	rm -rf ./models/test
	rm -rf ./logs/test
	rm -rf lightning_logs/

help:
	@echo "Available targets:"
	@echo "  view-training       - View training logs via TensorBoard"
	@echo "  train-encoder       - Train the encoder"
	@echo "  local-test-transformer       - Test the train-transformer pipeline locally"
	@echo "  local-test-encoder       - Test the train-encoder pipeline locally"
	@echo "  gcp-create       - Create a new GCP instance, per specifications"
	@echo "  gcp-start       - Spin up an existing GCP instance"
	@echo "  gcp-ssh       - SSH into an active GCP instance"
	@echo "  gcp-stop       - Spin down an active GCP instance"
	@echo "  clean       - Clean up local artifacts"
	@echo ""
	@echo "Configuration variables (can be overridden):"
	@echo "  GCP_ZONE=$(GCP_ZONE)"
	@echo "  VM_INSTANCE_NAME=$(VM_INSTANCE_NAME)"
	@echo "  MACHINE_TYPE=$(MACHINE_TYPE)"
	@echo "  DISK_SIZE=$(DISK_SIZE)"
	@echo "  DATASET_PATH=$(DATASET_PATH)"
	@echo "  MODEL=$(MODEL)"
	@echo ""

.PHONY: view-training train-encoder local-test-transformer local-test-encoder gcp-create gcp-start gcp-ssh gcp-stop clea n