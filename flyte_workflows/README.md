# ARC Encoder Flyte Deployment

This directory contains the infrastructure to train your ARC Encoder model on the cloud using Flyte.

## Overview

The Flyte deployment consists of:
- **Workflow Definition**: Defines data loading, training, and evaluation tasks
- **Docker Image**: Containerized environment with all dependencies
- **Configuration**: Cluster connection and resource settings
- **Scripts**: Helper scripts for deployment and management

## Quick Start

### 1. Prerequisites

Ensure you have:
- Flyte cluster access (or local Flyte installation)
- Docker with push access to a container registry
- Python 3.10+ with required dependencies

### 2. Setup

```bash
# Install dependencies
make install

# Make scripts executable
make setup
```

### 3. Local Testing

Test your training script locally first:
```bash
# Quick local test (2 epochs, small batch)
make test-local

# Test the Flyte workflow locally
make test-flyte
```

### 4. Configure for Your Environment

Edit the configuration files:
- `flyte_workflows/config.yaml` - For cloud deployment
- `flyte_workflows/config-local.yaml` - For local testing

Update these fields:
```yaml
admin:
  endpoint: flyte.your-cluster.com  # Your Flyte admin endpoint

storage:
  container: your-bucket-name  # Your S3/GCS bucket

execution:
  cluster: your-cluster-name  # Your cluster name
```

### 5. Deploy to Cloud

```bash
# Full deployment pipeline
make deploy REGISTRY=your-registry.com

# Or step by step:
make build REGISTRY=your-registry.com
make register
make launch
```

## File Structure

```
flyte_workflows/
├── arc_encoder_training.py  # Main workflow definition
├── config.yaml             # Cloud configuration
└── config-local.yaml       # Local testing configuration

scripts/
├── train_encoder.py         # Standalone training script
└── flyte_deploy.py         # Deployment management script

requirements-flyte.txt       # Flyte-specific dependencies
Dockerfile.flyte            # Container definition
Makefile                    # Convenient make targets
```

## Workflow Components

### Tasks

1. **load_and_prepare_data**: Loads ARC dataset and creates data loaders
2. **train_encoder_model**: Trains the model using PyTorch Lightning
3. **evaluate_model**: Evaluates trained model and returns metrics

### Resources

- **Data Loading**: 2-4 CPUs, 4-8GB RAM
- **Training**: 4-8 CPUs, 8-16GB RAM, 1 GPU
- **Evaluation**: 2 CPUs, 4GB RAM

## Usage Examples

### Basic Training

```bash
# Train with default parameters
make deploy REGISTRY=myregistry.com

# Custom training parameters
make deploy REGISTRY=myregistry.com \
  EPOCHS=50 \
  BATCH_SIZE=64 \
  LEARNING_RATE=5e-4
```

### Advanced Configuration

```bash
# Use different project/domain
make deploy REGISTRY=myregistry.com \
  PROJECT=arc-research \
  DOMAIN=production \
  EPOCHS=100

# Use different dataset
make deploy REGISTRY=myregistry.com \
  DATASET_PATH=evaluation
```

### Monitoring

```bash
# Check cluster status
make status

# View running executions
make executions

# View logs (use Flyte Console or CLI)
flytectl get execution-details <execution-id> --config flyte_workflows/config.yaml
```

## Customization

### Modify Training Parameters

Edit the `TrainingConfig` dataclass in `arc_encoder_training.py`:

```python
@dataclass
class TrainingConfig:
    epochs: int = 20  # Change default epochs
    batch_size: int = 64  # Change default batch size
    # ... other parameters
```

### Add New Tasks

Add new tasks to the workflow:

```python
@task(container_image=arc_training_image)
def my_new_task(input_data: FlyteFile) -> FlyteFile:
    # Your task implementation
    return output_file

@workflow
def enhanced_workflow(...):
    data = load_and_prepare_data(...)
    processed_data = my_new_task(data)  # Add your task
    model, logs = train_encoder_model(processed_data, ...)
    return model, logs
```

### Resource Requirements

Adjust resources in task decorators:

```python
@task(
    requests=Resources(cpu="8", mem="16Gi", gpu="2"),
    limits=Resources(cpu="16", mem="32Gi", gpu="2")
)
def intensive_task(...):
    # Your implementation
```

## Troubleshooting

### Common Issues

1. **Docker build failures**:
   - Ensure all dependencies are in `requirements-flyte.txt`
   - Check Dockerfile.flyte for correct base image

2. **Workflow registration fails**:
   - Verify Flyte cluster connectivity
   - Check project/domain permissions

3. **Task execution fails**:
   - Review resource requirements
   - Check input data availability
   - Examine task logs in Flyte Console

### Debugging

```bash
# Check workflow syntax
pyflyte check flyte_workflows/arc_encoder_training.py

# Local workflow execution with verbose logging
pyflyte run --config flyte_workflows/config-local.yaml \
  flyte_workflows/arc_encoder_training.py \
  arc_encoder_training_workflow \
  --verbose

# View detailed execution logs
flytectl get execution-details <execution-id> \
  --config flyte_workflows/config.yaml
```

## Performance Optimization

### Resource Allocation

- **CPU**: Start with 2-4 cores, scale based on data loading bottlenecks
- **Memory**: Monitor memory usage; increase if OOM errors occur
- **GPU**: Use single GPU for most training; scale to multi-GPU if needed
- **Storage**: Ensure sufficient storage for model checkpoints and logs

### Cost Optimization

- **Preemptible/Spot Instances**: Use for non-critical training
- **Auto-scaling**: Configure cluster auto-scaling for cost efficiency
- **Caching**: Enable task caching for faster iterations

## Integration with Existing Code

This deployment integrates seamlessly with your existing ARC codebase:

- Uses your existing `ARCEncoder` and `ARCDataClasses`
- Preserves your training configuration system
- Maintains compatibility with your data loading pipeline
- Supports your current model architecture

## Next Steps

1. **Set up monitoring**: Integrate with monitoring tools (Grafana, Prometheus)
2. **Add experiment tracking**: Use Weights & Biases or MLflow integration
3. **Implement model serving**: Add model deployment pipeline
4. **Scale training**: Experiment with distributed training for larger models
5. **Add data versioning**: Integrate with data versioning tools (DVC, Pachyderm)

For more advanced features, refer to the [Flyte documentation](https://docs.flyte.org/).