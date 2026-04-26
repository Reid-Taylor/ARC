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
	@echo "Training the ARC Encoder..."
	uv run scripts/train_encoder.py \
		--config train_config \
		--dataset-path $(DATASET_PATH) \
		--model-save-path ./models/encoder \
		--log-path ./logs/encoder \
		$(if $(EXPERIMENT_NAME),--experiment-name $(EXPERIMENT_NAME),)

local-test-encoder:
	@echo "Running local test of ARC Encoder training script..."
	uv run scripts/train_encoder.py \
		--config test_config \
		--dataset-path $(DATASET_PATH) \
		--model-save-path ./models/tmp \
		--log-path ./logs/tmp \
		$(if $(EXPERIMENT_NAME),--experiment-name $(EXPERIMENT_NAME),)

view-training:
	tensorboard --logdir logs/test

compile:
	@echo "Building Typst document..."
	typst compile documentation/documentation.typ

clean:
	@echo "Cleaning up local artifacts..."
	rm -rf ./models
	rm -rf ./logs
	rm -rf lightning_logs/

clean-tmp:
	@echo "Cleaning up temporary artifacts..."
	rm -rf ./models/tmp
	rm -rf ./logs/tmp
	rm -rf lightning_logs/tmp

help:
	@echo "Available targets:"
	@echo "  train-encoder       - Train the encoder with train_config"
	@echo "  local-test-encoder  - Test the encoder pipeline locally with test_config"
	@echo "  view-training       - View training logs via TensorBoard"
	@echo "  compile             - (Re)Build Typst documentation"
	@echo "  clean               - Clean up artifacts (models, logs)"
	@echo "  clean-tmp           - Clean up temporary artifacts only"
	@echo "  gcp-create          - Create a new GCP instance"
	@echo "  gcp-delete          - Delete a GCP instance"
	@echo "  gcp-start           - Start an existing GCP instance"
	@echo "  gcp-stop            - Stop an active GCP instance"
	@echo "  gcp-ssh             - SSH into an active GCP instance"
	@echo "  gcp-copy            - Copy models and logs from GCP instance"
	@echo "  gcp-copy-logs       - Copy only logs from GCP instance"
	@echo ""
	@echo "Configuration variables (can be overridden):"
	@echo "  GCP_ZONE=$(GCP_ZONE)"
	@echo "  VM_INSTANCE_NAME=$(VM_INSTANCE_NAME)"
	@echo "  MACHINE_TYPE=$(MACHINE_TYPE)"
	@echo "  DISK_SIZE=$(DISK_SIZE)"
	@echo "  DATASET_PATH=$(DATASET_PATH)"
	@echo "  EXPERIMENT_NAME=$(EXPERIMENT_NAME)"
	@echo ""

.PHONY: train-encoder local-test-encoder view-training compile clean clean-tmp gcp-create gcp-delete gcp-start gcp-stop gcp-ssh gcp-copy gcp-copy-logs help