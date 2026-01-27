#!/usr/bin/env python3
"""
Standalone training script for ARC Encoder that can be executed independently.
"""
import argparse
import sys
import os
import json
from pathlib import Path
from typing import Dict, Any

import torch
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger
from tensordict import TensorDict

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

device = get_device()
print(f"Using: {device}")
if device.type == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.arc.config_loader import load_config
from src.arc.ARCDataClasses import ARCProblemSet
from src.arc.ARCEncoder import MultiTaskEncoder


def create_dataloader(config: Dict[str, Any]):
    encoder_config = config['model']['encoder']
    dataset_path:str = config['training']['shared']['dataset_path']
    batch_size:int = config['training']['encoder']['batch_size']
    
    all_grids = ARCProblemSet.load_from_data_directory(dataset_path)['list_of_grids']
    num_samples = len(all_grids)
    
    def collate_fn(batch):
        names = [item["name"] for item in batch]

        padded_grids = torch.stack([item["padded_grid"] for item in batch], dim=0).reshape(-1, encoder_config['grid_size'])
        encoded_grids = torch.stack([item["encoded_grid"] for item in batch], dim=0).reshape(-1, encoder_config['grid_size'])

        # Augmentation presence flags
        roll_augmentations = torch.stack([torch.tensor("roll" in item["augmentation_set"], dtype=torch.bool) for item in batch], dim=0).reshape(-1,1).float()
        scale_grid_augmentations = torch.stack([torch.tensor("scale_grid" in item["augmentation_set"], dtype=torch.bool) for item in batch], dim=0).reshape(-1,1).float()
        isolate_color_augmentations = torch.stack([torch.tensor("isolate_color" in item["augmentation_set"], dtype=torch.bool) for item in batch], dim=0).reshape(-1,1).float()

        padded_augmented_grids = torch.stack([item["padded_augmented_grid"] for item in batch], dim=0).reshape(-1, encoder_config['grid_size'])
        encoded_augmented_grids = torch.stack([item["encoded_augmented_grid"] for item in batch], dim=0).reshape(-1, encoder_config['grid_size'])

        # Meta attributes
        area = torch.stack([torch.tensor(item["meta"].area, dtype=torch.float32) for item in batch], dim=0).reshape(-1,1)
        grid_size = torch.stack([torch.tensor(item["meta"].grid_size, dtype=torch.float32) for item in batch], dim=0).reshape(-1,2)
        num_colors = torch.stack([torch.tensor(item["meta"].num_colors, dtype=torch.float32) for item in batch], dim=0)
        color_map = torch.stack([torch.tensor(item["meta"].color_map, dtype=torch.float32) for item in batch], dim=0).reshape(-1,10)

        return TensorDict(
            {
                "name": names,

                "padded_grid": padded_grids,
                "encoded_grid": encoded_grids,
                "padded_augmented_grid": padded_augmented_grids,
                "encoded_augmented_grid": encoded_augmented_grids,
                "predicted_grid": None,
                "attended_dict":None,

                "area": area,
                "grid_size": grid_size,
                "num_colors": num_colors,
                "color_map": color_map,
                
                "predicted_area": None,
                "predicted_grid_size": None,
                "predicted_num_colors": None,
                "predicted_color_map": None,

                "presence_roll": roll_augmentations,
                "presence_scale_grid": scale_grid_augmentations,
                "presence_isolate_color": isolate_color_augmentations,
                
                "predicted_presence_roll": None,
                "predicted_presence_scale_grid": None,
                "predicted_presence_isolate_color": None,

                "online_embedding": None,
                "target_embedding": None,
                "online_representation": None,
                "target_representation": None,
                "predicted_target_representation": None
            },
            batch_size=len(batch),
            device=get_device()
        )
    
    # Create dataloader
    train_dataloader = torch.utils.data.DataLoader(
        all_grids[:int(0.9*num_samples)],
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=2 if torch.cuda.is_available() else 0,
        pin_memory=(get_device().type=="cuda")
    )
    # Create dataloader
    val_dataloader = torch.utils.data.DataLoader(
        all_grids[int(0.9*num_samples):],
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=2 if torch.cuda.is_available() else 0,
        pin_memory=(get_device().type=="cuda")
    )
    
    return train_dataloader, val_dataloader


def create_model(config: Dict[str, Any]) -> MultiTaskEncoder:
    """Create and initialize the model."""
    encoder_config = config['model']['encoder']
    downstream_attributes_config = config['model']['encoder']['downstream_attributes']
    contrastive_attributes_config = config['model']['encoder']['contrastive_attributes']
    shared_model_config = config['model']['shared']
    learning_rate: float = config['model']['encoder']['learning_rate']
    alpha: float = config['model']['encoder']['alpha']
    
    model = MultiTaskEncoder(
        attribute_requirements=list(downstream_attributes_config.keys()),
        task_type={
            key: contrastive_attributes_config[key]['task_type'] 
            for key in contrastive_attributes_config.keys()
        },
        learning_rate=learning_rate,
        alpha=alpha,
        tau=config['model']['encoder']['tau'],
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
                "output_size": shared_model_config['latent_size'],
                "activation": "identity"
            },
            "Attribute Detector": {
                key: {
                    "input_size": shared_model_config['latent_size'],
                    "output_size": 1,
                    "activation": contrastive_attributes_config[key].get('activation', "sigmoid")
                } for key in contrastive_attributes_config.keys() 
                if contrastive_attributes_config[key].get('task_type', None) == 'task_sensitive'
            },
            "Attribute Predictor": {
                key: {
                    "input_size": shared_model_config['latent_size'],
                    "hidden_size": downstream_attributes_config[key]['hidden_size'],
                    "output_size": downstream_attributes_config[key]['output_size']
                } for key in downstream_attributes_config.keys()
            },
        }
    ).to(get_device())
    
    return model


def setup_trainer(
    epochs: int, 
    model_save_path: str, 
    log_path: str, 
    use_gpu: bool = True
) -> L.Trainer:
    """Setup Lightning trainer with callbacks and logger."""
    
    # Create directories if they don't exist
    os.makedirs(model_save_path, exist_ok=True)
    os.makedirs(log_path, exist_ok=True)
    
    # Setup callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=model_save_path,
        filename="arc_encoder_{epoch:02d}_{val_loss:.2f}",
        monitor="val_loss",
        mode="min",
        save_top_k=3,
        save_last=True
    )
    
    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=4,
        mode="min"
    )
    
    # Setup logger
    logger = TensorBoardLogger(
        save_dir=log_path,
        name="arc_encoder",
        version=f"train_{torch.randint(0, 10000, (1,)).item()}"
    )
    
    # Determine accelerator
    accelerator = "gpu" if use_gpu and torch.cuda.is_available() else "cpu"
    devices = 1 if accelerator == "gpu" else "auto"
    
    # Initialize trainer
    trainer = L.Trainer(
        max_epochs=epochs,
        callbacks=[checkpoint_callback, early_stopping],
        logger=logger,
        accelerator=accelerator,
        devices=devices,
        log_every_n_steps=1,
        # val_check_interval=0.25,
        check_val_every_n_epoch=10,
        enable_progress_bar=True,
        enable_model_summary=True
    )
    
    return trainer


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train ARC Encoder model")
    parser.add_argument("--config", type=str, help="Name of config file")
    parser.add_argument("--dataset-path", type=str, default="training", help="Dataset path")
    parser.add_argument("--model-save-path", type=str, default="./models/encoder", help="Model save path")
    parser.add_argument("--log-path", type=str, default="./lightning_logs", help="Logging path")
    parser.add_argument("--no-gpu", action="store_true", help="Disable GPU usage")
    parser.add_argument("--output-metrics", type=str, help="Path to save training metrics JSON")
    
    args = parser.parse_args()
    
    # Load configuration
    if args.config:
        config = load_config(args.config)
    else:
        config = load_config()
    
    print(f"Starting ARC Encoder training with the following parameters:")
    print(f"  Dataset path: {args.dataset_path}")
    print(f"  Model save path: {args.model_save_path}")
    print(f"  Using GPU: {not args.no_gpu and torch.cuda.is_available()}")
    
    # Create dataloader
    print("Loading dataset...")
    train_dataloader, val_dataloaders = create_dataloader(config)
    print(f"Dataset loaded with {len(train_dataloader)} batches")
    
    # Create model
    print("Initializing model...")
    model = create_model(config)
    print(f"Model initialized with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Setup trainer
    print("Setting up trainer...")
    trainer = setup_trainer(
        config['training']['encoder']['epochs'],
        args.model_save_path, 
        args.log_path, 
        use_gpu=not args.no_gpu
    )
    
    # Train model
    print("Starting training...")
    trainer.fit(model=model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloaders)
    
    # Save training metrics
    if args.output_metrics:
        metrics = {
            "best_checkpoint": trainer.checkpoint_callback.best_model_path,
            "best_score": trainer.checkpoint_callback.best_model_score.item() if trainer.checkpoint_callback.best_model_score else None,
            "total_epochs": trainer.current_epoch,
            "num_parameters": sum(p.numel() for p in model.parameters())
        }
        
        os.makedirs(os.path.dirname(args.output_metrics), exist_ok=True)
        with open(args.output_metrics, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        print(f"Training metrics saved to {args.output_metrics}")
    
    print("Training completed!")
    print(f"Best checkpoint: {trainer.checkpoint_callback.best_model_path}")
    

if __name__ == "__main__":
    main()