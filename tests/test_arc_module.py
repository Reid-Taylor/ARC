import pytest
import torch
import numpy as np
import lightning as L
import sys
from pathlib import Path
from tensordict import TensorDict

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

print("Current working directory:", Path.cwd())
print("Project root:", project_root)

# Now import with correct paths
from src.arc.config_loader import load_config, parse_args
from src.arc.ARCDataClasses import ARCProblemSet
from src.arc.ARCModule import MultiTaskEncoder, positional_encodings

class TestARCModule:
    @pytest.fixture
    def config(self):
        """Load test configuration."""
        return load_config()
    
    @pytest.fixture
    def model(self, config):
        """Create model instance."""
        model_config = config['model']
        contrastive_attributes = config.get('contrastive_attributes', {})
        downstream_attributes = config.get('downstream_attributes', {})
        
        return MultiTaskEncoder(
            attribute_requirements = list(downstream_attributes.keys()),
            task_type = {
                key: contrastive_attributes[key]['task_type'] for key in contrastive_attributes.keys()
            },
            learning_rate=model_config['learning_rate'],
            alpha=model_config['alpha'],
            **{
                "Encoder": {
                    "input_size": model_config['grid_size'],
                    "attention_sizes": model_config['attention_sizes'],
                    "output_size": model_config['latent_size']
                },
                "Decoder": {
                    "input_size": model_config['latent_size'],
                    "attention_sizes": model_config['attention_sizes'],
                    "output_size": model_config['grid_size']
                },
                "Contrastive Projection": { #TODO: Update this to a FC layer if needed (experiment)
                    "input_size": model_config['latent_size'],
                    "hidden_size": model_config['latent_size'],
                    "output_size": model_config['latent_size']
                },
                "Contrastive Predictor": { 
                    "input_size": model_config['latent_size'],
                    "output_size": model_config['latent_size']
                },
                "Attribute Detector": { #NOTE: This is for sensitive-attribute detection heads
                    key: {
                        "input_size": model_config['latent_size'],
                        "output_size": 1,
                        "activation": contrastive_attributes[key].get('activation', "sigmoid")
                    } for key in contrastive_attributes.keys() 
                    if contrastive_attributes[key].get('task_type', None) == 'task_sensitive'
                },
                "Attribute Head": {
                    key: {
                        "input_size": model_config['latent_size'],
                        "hidden_size": downstream_attributes[key]['hidden_size'],
                        "output_size": downstream_attributes[key]['output_size']
                    } for key in downstream_attributes.keys()
                },
            }
        )
    
    @pytest.fixture
    def dataloader(self, config):
        """Create data loader."""
        training_config = config['training']
        model_config = config['model']
        
        dataset = ARCProblemSet.load_from_data_directory(training_config['dataset_path'])
        
        def collate_fn(batch):
            names = [item["name"] for item in batch]
            padded_grids = torch.stack([item["padded_grid"] for item in batch], dim=0).reshape(-1, model_config['grid_size'])
            return TensorDict(
                {
                    "name": names,
                    "embedding": None,
                    "padded_grid": padded_grids,
                    "encoded_grid": padded_grids + positional_encodings.reshape(-1, model_config['grid_size']),
                    "predicted_grid": None,
                    "area": [item["meta"].area for item in batch],
                    "predicted_area": None,
                    "grid_size": [item["meta"].grid_size for item in batch],
                    "predicted_grid_size": None,
                    "num_colors": [item["meta"].num_colors for item in batch],
                    "predicted_num_colors": None,
                    "color_map": [item["meta"].color_map for item in batch],
                    "predicted_color_map": None
                },
                batch_size=len(batch)
            )
        
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=training_config['batch_size'],
            shuffle=True,
            collate_fn=collate_fn
        )
    
    def test_model_training(self, model, dataloader, config):
        """Test model training process."""
        trainer = L.Trainer(max_epochs=config['training']['epochs'])
        trainer.fit(model=model, train_dataloaders=dataloader)
        
        # Basic assertions
        assert model.training == False  # Should be in eval mode after training
    
    def test_model_inference(self, model, dataloader, config):
        """Test model inference and output visualization."""
        sample_batch = next(iter(dataloader))
        model.eval()
        
        with torch.no_grad():
            # Forward pass
            encoder_out = model.encoder(sample_batch)
            sample_batch["embedding"] = encoder_out["embedding"]
            decoder_out = model.decoder(sample_batch)
            reconstructed = decoder_out["predicted_grid"]
            
            # Attribute predictions
            attribute_predictions = {}
            for attr_key in config['attributes'].keys():
                attr_head = getattr(model, f"attribute_head_{attr_key}")
                attr_out = attr_head(sample_batch)
                attribute_predictions[attr_key] = attr_out[f"predicted_{attr_key}"]
            
            # Assertions
            assert reconstructed.shape[0] == sample_batch.batch_size
            assert len(attribute_predictions) == len(config['attributes'])
            
            # Optional: Print results for manual inspection
            self._print_results(sample_batch, reconstructed, attribute_predictions, config)
    
    def _print_results(self, sample_batch, reconstructed, attribute_predictions, config):
        """Print reconstruction and attribute prediction results."""
        sample_idx = config['testing']['sample_idx']
        grid_size = config['testing']['grid_display_size']
        
        sample_grid_size = sample_batch["grid_size"][sample_idx].squeeze().to(torch.int32)
        original = sample_batch["padded_grid"][sample_idx].reshape(grid_size, grid_size)
        prediction = reconstructed[sample_idx].reshape(grid_size, grid_size)
        
        print("=== GRID RECONSTRUCTION ===")
        print("Original grid:\n", original[0:sample_grid_size[0].item(), 0:sample_grid_size[1].item()].cpu().numpy())
        print("Predicted grid:\n", np.round(prediction[0:sample_grid_size[0].item(), 0:sample_grid_size[1].item()].cpu().numpy()))
        
        print("\n=== ATTRIBUTE PREDICTIONS ===")
        for attr_key in config['attributes'].keys():
            predicted = attribute_predictions[attr_key][sample_idx].cpu().numpy()
            actual = sample_batch[attr_key][sample_idx]
            if hasattr(actual, 'cpu'):
                actual = actual.cpu().numpy()
            
            print(f"\n{attr_key.upper()}:")
            print(f"  Predicted: {predicted}")
            print(f"  Actual:    {actual}")


def run_interactive_test():
    """Run interactive test with command line arguments."""
    args = parse_args()
    config = load_config(args.config)
    
    # Override config with command line args if provided
    if args.epochs:
        config['training']['epochs'] = args.epochs
    if args.batch_size:
        config['training']['batch_size'] = args.batch_size
    if args.sample_idx:
        config['testing']['sample_idx'] = args.sample_idx
    
    # Create test instance and run
    test_instance = TestARCModule()
    model = test_instance.model(config)
    dataloader = test_instance.dataloader(config)
    
    if args.mode in ['train', 'both']:
        print("Starting training...")
        test_instance.test_model_training(model, dataloader, config)
    
    if args.mode in ['test', 'both']:
        print("Running inference test...")
        test_instance.test_model_inference(model, dataloader, config)


if __name__ == "__main__":
    run_interactive_test()