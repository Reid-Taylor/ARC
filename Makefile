# GCP parameters
GCP_ZONE ?= us-central1-f
VM_INSTANCE_NAME ?= instance-101
MACHINE_TYPE ?= e2-highmem-16
DISK_SIZE ?= 120GB

# Training parameters
DATASET_PATH ?= training
MODEL ?= encoder
EXPERIMENT_NAME ?=

gcp-create:
	@echo "Creating GCP instance..."
	gcloud compute instances create $(VM_INSTANCE_NAME) \
		--project amplified-hull-484821-b5 \
		--zone=$(GCP_ZONE) \
		--machine-type $(MACHINE_TYPE) \
		--boot-disk-size $(DISK_SIZE) 

gcp-delete:
	@echo "Deleting GCP instance..."
	gcloud compute instances delete $(VM_INSTANCE_NAME) --zone=$(GCP_ZONE)

gcp-start:
	gcloud compute instances start $(VM_INSTANCE_NAME) --zone=$(GCP_ZONE)

gcp-stop:
	gcloud compute instances stop $(VM_INSTANCE_NAME) --zone=$(GCP_ZONE)

gcp-ssh:
	gcloud compute ssh $(VM_INSTANCE_NAME) --zone=$(GCP_ZONE)

gcp-copy-logs:
	gcloud compute scp --zone=$(GCP_ZONE) --recurse $(VM_INSTANCE_NAME):/home/reidtaylor/ARC/logs/test ./logs

gcp-copy:
	gcloud compute scp --zone=$(GCP_ZONE) --recurse $(VM_INSTANCE_NAME):/home/reidtaylor/ARC/models/test ./models/test
	gcloud compute scp --zone=$(GCP_ZONE) --recurse $(VM_INSTANCE_NAME):/home/reidtaylor/ARC/logs/test ./logs

train-encoder:
	@echo "Running local test of ARC Encoder training..."
	uv run scripts/train_contrastive_encoder.py \
		--config train_config \
		--dataset-path $(DATASET_PATH) \
		--model-save-path ./models/test/contrastive_encoder \
		--log-path ./logs/test/contrastive_encoder \
		$(if $(EXPERIMENT_NAME),--experiment-name $(EXPERIMENT_NAME),)

local-test-encoder:
	@echo "Running local test of ARC Encoder training..."
	uv run scripts/train_contrastive_encoder.py \
		--config test_config \
		--dataset-path $(DATASET_PATH) \
		--model-save-path ./models/test/tmp/contrastive_encoder \
		--log-path ./logs/test/tmp/contrastive_encoder \
		$(if $(EXPERIMENT_NAME),--experiment-name $(EXPERIMENT_NAME),)

view-training:
	tensorboard --logdir logs/test

compile:
	@echo "Building Typst document..."
	typst compile documentation/documentation.typ

clean:
	@echo "Cleaning up local artifacts..."
	rm -rf ./models/test
	rm -rf ./logs/test
	rm -rf lightning_logs/

clean-tmp:
	@echo "Cleaning up temporary artifacts..."
	rm -rf ./models/test/tmp
	rm -rf ./logs/test/tmp
	rm -rf lightning_logs/tmp

help:
	@echo "Available targets:"
	@echo "  view-training       - View training logs via TensorBoard"
	@echo "  train-encoder       - Train the encoder"
	@echo "  local-test-encoder       - Test the train-encoder pipeline locally"
	@echo "  gcp-create       - Create a new GCP instance, per specifications"
	@echo "  gcp-delete       - Delete a GCP instance"
	@echo "  gcp-start       - Spin up an existing GCP instance"
	@echo "  gcp-ssh       - SSH into an active GCP instance"
	@echo "  gcp-stop       - Spin down an active GCP instance"
	@echo "  clean       - Clean up local artifacts"
	@echo "  clean-tmp       - Clean up temporary artifacts"
	@echo ""
	@echo "Configuration variables (can be overridden):"
	@echo "  GCP_ZONE=$(GCP_ZONE)"
	@echo "  VM_INSTANCE_NAME=$(VM_INSTANCE_NAME)"
	@echo "  MACHINE_TYPE=$(MACHINE_TYPE)"
	@echo "  DISK_SIZE=$(DISK_SIZE)"
	@echo "  DATASET_PATH=$(DATASET_PATH)"
	@echo "  MODEL=$(MODEL)"
	@echo ""

.PHONY: view-training train-encoder local-test-transformer local-test-encoder gcp-create gcp-delete gcp-start gcp-ssh gcp-stop clean