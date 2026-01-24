#!/usr/bin/env python3
"""
Standalone training script for ARC Encoder that can be executed independently
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

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.arc.config_loader import load_config
from src.arc.ARCDataClasses import ARCProblemSet
from src.arc.ARCTransformer import TransformationDescriber

def create_dataloader(config: Dict[str, Any], batch_size: int, dataset_path: str):
    training_config = config['training']['transformer']
    shared_training_config = config['training']['shared']
    
    all_tensordicts = ARCProblemSet.load_from_data_directory(shared_training_config['dataset_path'])['list_of_tensordicts']
    
    def collate_fn(batch):
        names = [item["problem_name"] for item in batch]
        num_examples = torch.stack([item["num_examples"] for item in batch])
        examples = torch.stack([item["examples"] for item in batch])
        challenge = torch.stack([item["challenge"] for item in batch])
        solution = torch.stack([item["solution"] for item in batch])
        
        return TensorDict(
            {
                "name": names,
                "num_examples": num_examples,
                "examples": examples,
                "challenge": challenge,
                "solution": solution,
                "transformation_description":None,
                "random_description":None
            },
            batch_size=len(batch)
        )
    
    return torch.utils.data.DataLoader(
        all_tensordicts,
        batch_size=training_config['batch_size'],
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0
    )
    

def create_model(config: Dict[str, Any], learning_rate: float, alpha: float) -> TransformationDescriber:
    """Create and initialize the model."""
    shared_model_config = config['model']['shared']
    
    model = TransformationDescriber(
        learning_rate=learning_rate,
        alpha=alpha,
        **{
            "TransformationDescriber": {
                "input_size": shared_model_config['latent_size'],
                "output_size": shared_model_config['transformation_dimension_size']
            }
        }
    )
    
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
        filename="arc_transformer_{epoch:02d}_{val_loss:.2f}",
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
        save_dir=log_path,
        name="arc_transformer",
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
        log_every_n_steps=10,
        val_check_interval=0.25,
        enable_progress_bar=True,
        enable_model_summary=True
    )
    
    return trainer


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train ARC Transformer model")
    parser.add_argument("--config", type=str, help="Path to config file")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--alpha", type=float, default=0.85, help="Alpha parameter for multi-task learning")
    parser.add_argument("--dataset-path", type=str, default="training", help="Dataset path")
    parser.add_argument("--model-save-path", type=str, default="./models/transformer", help="Model save path")
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
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.learning_rate}")
    print(f"  Alpha: {args.alpha}")
    print(f"  Dataset path: {args.dataset_path}")
    print(f"  Model save path: {args.model_save_path}")
    print(f"  Using GPU: {not args.no_gpu and torch.cuda.is_available()}")
    
    # Create dataloader
    print("Loading dataset...")
    dataloader = create_dataloader(config, args.batch_size, args.dataset_path)
    print(f"Dataset loaded with {len(dataloader)} batches")
    
    # Create model
    print("Initializing model...")
    model = create_model(config, args.learning_rate, args.alpha)
    print(f"Model initialized with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Setup trainer
    print("Setting up trainer...")
    trainer = setup_trainer(
        args.epochs, 
        args.model_save_path, 
        args.log_path, 
        use_gpu=not args.no_gpu
    )
    
    # Train model
    print("Starting training...")
    trainer.fit(model=model, train_dataloaders=dataloader)
    
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