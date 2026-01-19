"""
Flyte workflow for training ARC Encoder model on the cloud.

This workflow orchestrates:
1. Data loading and preprocessing
2. Model initialization and training
3. Model artifact saving and metrics logging
"""
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any
import os
import torch
import lightning as L
from flytekit import task, workflow, Resources, ImageSpec
from flytekit.types.file import FlyteFile
from flytekit.types.directory import FlyteDirectory
from flytekitplugins.pytorch import PyTorch

# Custom image specification for the workflow
arc_training_image = ImageSpec(
    name="arc-encoder-training",
    python_version="3.10",
    registry="your-registry.com",  # Replace with your Docker registry
    requirements="requirements-flyte.txt",
    base_image="pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime"
)

@dataclass
class TrainingConfig:
    """Configuration for training parameters."""
    epochs: int = 10
    batch_size: int = 32
    learning_rate: float = 1e-3
    alpha: float = 0.85
    dataset_path: str = "training"
    model_save_path: str = "/tmp/arc_encoder_model"
    log_path: str = "/tmp/lightning_logs"


@task(
    container_image=arc_training_image,
    requests=Resources(cpu="2", mem="4Gi", storage="10Gi"),
    limits=Resources(cpu="4", mem="8Gi", storage="20Gi"),
    cache=True,
    cache_version="1.0"
)
def load_and_prepare_data(dataset_path: str, batch_size: int) -> FlyteDirectory:
    """
    Load ARC dataset and prepare data loaders.
    
    Returns a directory containing serialized dataloaders.
    """
    import sys
    import torch.utils.data
    from src.arc.ARCDataClasses import ARCProblemSet
    from src.arc.config_loader import load_config
    import pickle
    import os
    
    # Load configuration
    config = load_config()
    
    # Load dataset
    all_grids = ARCProblemSet.load_from_data_directory(dataset_path)['list_of_grids']
    
    def collate_fn(batch):
        """Collate function for dataloader."""
        import torch
        from jaxtyping import Float
        
        # Create batch tensors
        batch_dict = {
            "padded_grid": torch.stack([grid.padded_grid for grid in batch]).squeeze(1),
            "padded_augmented_grid": torch.stack([grid.padded_augmented_grid for grid in batch]).squeeze(1),
            "attributes": torch.stack([grid.attributes for grid in batch]).squeeze(1),
            "area": torch.stack([grid.meta.area for grid in batch]),
            "grid_size": torch.stack([grid.meta.grid_size for grid in batch]).squeeze(1),
            "num_colors": torch.stack([grid.meta.num_colors for grid in batch]),
            "color_map": torch.stack([grid.meta.color_map for grid in batch]),
        }
        
        return batch_dict
    
    # Create dataloader
    dataloader = torch.utils.data.DataLoader(
        all_grids,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=2
    )
    
    # Create output directory
    output_dir = "/tmp/data_output"
    os.makedirs(output_dir, exist_ok=True)
    
    # Save dataloader and config
    with open(os.path.join(output_dir, "dataloader.pkl"), "wb") as f:
        pickle.dump(dataloader, f)
    
    with open(os.path.join(output_dir, "config.pkl"), "wb") as f:
        pickle.dump(config, f)
    
    return FlyteDirectory(path=output_dir)


@task(
    container_image=arc_training_image,
    requests=Resources(cpu="4", mem="8Gi", gpu="1", storage="20Gi"),
    limits=Resources(cpu="8", mem="16Gi", gpu="1", storage="40Gi"),
    task_config=PyTorch(worker=1)
)
def train_encoder_model(
    data_dir: FlyteDirectory, 
    training_config: TrainingConfig
) -> Tuple[FlyteFile, FlyteDirectory]:
    """
    Train the ARC Encoder model.
    
    Args:
        data_dir: Directory containing prepared data
        training_config: Training configuration parameters
    
    Returns:
        Tuple of (model checkpoint file, logs directory)
    """
    import sys
    import os
    import pickle
    import lightning as L
    from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
    from lightning.pytorch.loggers import TensorBoardLogger
    from src.arc.ARCEncoder import MultiTaskEncoder
    
    # Load data and config
    data_path = data_dir.download()
    
    with open(os.path.join(data_path, "dataloader.pkl"), "rb") as f:
        dataloader = pickle.load(f)
    
    with open(os.path.join(data_path, "config.pkl"), "rb") as f:
        config = pickle.load(f)
    
    # Extract model configuration
    encoder_config = config['model']['encoder']
    downstream_attributes_config = config['model']['encoder']['downstream_attributes']
    contrastive_attributes_config = config['model']['encoder']['contrastive_attributes']
    shared_model_config = config['model']['shared']
    
    # Initialize model
    model = MultiTaskEncoder(
        attribute_requirements=list(downstream_attributes_config.keys()),
        task_type={
            key: contrastive_attributes_config[key]['task_type'] 
            for key in contrastive_attributes_config.keys()
        },
        learning_rate=training_config.learning_rate,
        alpha=training_config.alpha,
        **{
            "Encoder": {
                "input_size": encoder_config['grid_size'],
                "attention_sizes": encoder_config['attention_sizes'],
                "output_size": shared_model_config['latent_size']
            },
            "Decoder": {
                "input_size": shared_model_config['latent_size'],
                "attention_sizes": encoder_config['attention_sizes'],
                "output_size": encoder_config['grid_size']
            },
            "Contrastive Projection": {
                "input_size": shared_model_config['latent_size'],
                "hidden_size": shared_model_config['latent_size'],
                "output_size": shared_model_config['latent_size']
            },
            "Contrastive Predictor": {
                "input_size": shared_model_config['latent_size'],
                "output_size": shared_model_config['latent_size']
            },
            "Attribute Detector": {
                key: {
                    "input_size": shared_model_config['latent_size'],
                    "output_size": 1,
                    "activation": contrastive_attributes_config[key].get('activation', "sigmoid")
                } for key in contrastive_attributes_config.keys() 
                if contrastive_attributes_config[key].get('task_type', None) == 'task_sensitive'
            },
            "Attribute Head": {
                key: {
                    "input_size": shared_model_config['latent_size'],
                    "hidden_size": downstream_attributes_config[key]['hidden_size'],
                    "output_size": downstream_attributes_config[key]['output_size']
                } for key in downstream_attributes_config.keys()
            },
        }
    )
    
    # Setup callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=training_config.model_save_path,
        filename="arc_encoder_{epoch:02d}_{val_loss:.2f}",
        monitor="val_loss",
        mode="min",
        save_top_k=3,
        save_last=True
    )
    
    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=5,
        mode="min"
    )
    
    # Setup logger
    logger = TensorBoardLogger(
        save_dir=training_config.log_path,
        name="arc_encoder",
        version="cloud_training"
    )
    
    # Initialize trainer
    trainer = L.Trainer(
        max_epochs=training_config.epochs,
        callbacks=[checkpoint_callback, early_stopping],
        logger=logger,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        log_every_n_steps=10,
        val_check_interval=0.25
    )
    
    # Train model
    trainer.fit(model=model, train_dataloaders=dataloader)
    
    # Get best checkpoint
    best_checkpoint = checkpoint_callback.best_model_path
    
    return (
        FlyteFile(path=best_checkpoint),
        FlyteDirectory(path=training_config.log_path)
    )


@task(
    container_image=arc_training_image,
    requests=Resources(cpu="2", mem="4Gi", storage="10Gi")
)
def evaluate_model(
    model_checkpoint: FlyteFile,
    data_dir: FlyteDirectory
) -> Dict[str, float]:
    """
    Evaluate the trained model and return metrics.
    
    Args:
        model_checkpoint: Path to trained model checkpoint
        data_dir: Directory containing test data
    
    Returns:
        Dictionary of evaluation metrics
    """
    import torch
    import pickle
    from src.arc.ARCEncoder import MultiTaskEncoder
    
    # Load model
    checkpoint_path = model_checkpoint.download()
    model = MultiTaskEncoder.load_from_checkpoint(checkpoint_path)
    model.eval()
    
    # Load test data
    data_path = data_dir.download()
    with open(os.path.join(data_path, "dataloader.pkl"), "rb") as f:
        dataloader = pickle.load(f)
    
    # Run evaluation
    metrics = {}
    total_loss = 0
    num_batches = 0
    
    with torch.no_grad():
        for batch in dataloader:
            # Get model predictions
            encoder_out = model.online_encoder(batch)
            batch["online_embedding"] = encoder_out["online_embedding"]
            decoder_out = model.decoder(batch)
            
            # Calculate reconstruction loss
            reconstruction_loss = torch.nn.functional.mse_loss(
                decoder_out["predicted_grid"], 
                batch["padded_grid"]
            )
            
            total_loss += reconstruction_loss.item()
            num_batches += 1
            
            # Break after a reasonable number of batches for evaluation
            if num_batches >= 10:
                break
    
    metrics["avg_reconstruction_loss"] = total_loss / num_batches
    metrics["num_parameters"] = sum(p.numel() for p in model.parameters())
    
    return metrics


@workflow
def arc_encoder_training_workflow(
    dataset_path: str = "training",
    epochs: int = 10,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    alpha: float = 0.85
) -> Tuple[FlyteFile, FlyteDirectory, Dict[str, float]]:
    """
    Complete workflow for training ARC Encoder model.
    
    Args:
        dataset_path: Path to dataset
        epochs: Number of training epochs
        batch_size: Training batch size
        learning_rate: Learning rate for optimization
        alpha: Alpha parameter for multi-task learning
    
    Returns:
        Tuple of (best model checkpoint, training logs, evaluation metrics)
    """
    # Create training config
    training_config = TrainingConfig(
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        alpha=alpha,
        dataset_path=dataset_path
    )
    
    # Load and prepare data
    data_dir = load_and_prepare_data(
        dataset_path=dataset_path,
        batch_size=batch_size
    )
    
    # Train model
    model_checkpoint, training_logs = train_encoder_model(
        data_dir=data_dir,
        training_config=training_config
    )
    
    # Evaluate model
    metrics = evaluate_model(
        model_checkpoint=model_checkpoint,
        data_dir=data_dir
    )
    
    return model_checkpoint, training_logs, metrics