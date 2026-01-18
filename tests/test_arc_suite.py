import pytest
import torch
import numpy as np
import lightning as L
import sys
from pathlib import Path
from tensordict import TensorDict

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.arc.config_loader import load_config, parse_args
from src.arc.ARCDataClasses import ARCProblemSet
from src.arc.ARCEncoder import MultiTaskEncoder, positional_encodings
from src.arc.ARCTransformer import TransformationDescriber
# from arc.ARCLinker import Linker

class TestARCEncoder:
    @pytest.fixture
    def config(self):
        """Load test configuration."""
        return load_config()
    
    @pytest.fixture
    def model(self, config):
        """Create model instance."""
        encoder_config = config['model']['encoder']
        downstream_attributes_config = config['model']['encoder']['downstream_attributes']
        contrastive_attributes_config = config['model']['encoder']['contrastive_attributes']
        shared_model_config = config['model']['shared']
        
        return MultiTaskEncoder(
            attribute_requirements = list(downstream_attributes_config.keys()),
            task_type = {
                key: contrastive_attributes_config[key]['task_type'] for key in contrastive_attributes_config.keys()
            },
            learning_rate=encoder_config['learning_rate'],
            alpha=encoder_config['alpha'],
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
                "Contrastive Projection": { #TODO: Update this to a FC layer if needed (experiment)
                    "input_size": shared_model_config['latent_size'],
                    "hidden_size": shared_model_config['latent_size'],
                    "output_size": shared_model_config['latent_size']
                },
                "Contrastive Predictor": { 
                    "input_size": shared_model_config['latent_size'],
                    "output_size": shared_model_config['latent_size']
                },
                "Attribute Detector": { #NOTE: This is for sensitive-attribute detection heads
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
    
    @pytest.fixture
    def dataloader(self, config):
        """Create data loader."""
        shared_training_config = config['training']['shared']
        encoder_training_config = config['training']['encoder']
        encoder_config = config['model']['encoder']
        
        all_grids = ARCProblemSet.load_from_data_directory(shared_training_config['dataset_path'])['list_of_grids']
        
        def collate_fn(batch):
            names = [item["name"] for item in batch]

            padded_grids = torch.stack([item["padded_grid"] for item in batch], dim=0).reshape(-1, encoder_config['grid_size'])

            roll_augmentations = torch.stack([torch.tensor("roll" in item["augmentation_set"], dtype=torch.bool) for item in batch], dim=0).reshape(-1,1).float()
            scale_grid_augmentations = torch.stack([torch.tensor("scale_grid" in item["augmentation_set"], dtype=torch.bool) for item in batch], dim=0).reshape(-1,1).float()
            isolate_color_augmentations = torch.stack([torch.tensor("isolate_color" in item["augmentation_set"], dtype=torch.bool) for item in batch], dim=0).reshape(-1,1).float()

            augmented_grids = torch.stack([item["augmented_grid"] for item in batch], dim=0).reshape(-1, encoder_config['grid_size'])

            area = torch.stack([torch.tensor(item["meta"].area, dtype=torch.float32) for item in batch], dim=0).reshape(-1,1)

            grid_size = torch.stack([torch.tensor(item["meta"].grid_size, dtype=torch.float32) for item in batch], dim=0).reshape(-1,2)

            num_colors = torch.stack([torch.tensor(item["meta"].num_colors, dtype=torch.float32) for item in batch], dim=0)

            color_map = torch.stack([torch.tensor(item["meta"].color_map, dtype=torch.float32) for item in batch], dim=0).reshape(-1,10)
            
            return TensorDict(
                {
                    "name": names,

                    "padded_grid": padded_grids,
                    "encoded_grid": padded_grids + positional_encodings.reshape(-1, encoder_config['grid_size']),
                    "augmented_grid": augmented_grids,
                    "predicted_grid": None,

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
                batch_size=len(batch)
            )
        
        return torch.utils.data.DataLoader(
            all_grids,
            batch_size=encoder_training_config['batch_size'],
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=0
        )
    
    def test_model_training(self, model, dataloader, config):
        """Test model training process."""
        trainer = L.Trainer(max_epochs=config['training']['encoder']['epochs'])
        trainer.fit(model=model, train_dataloaders=dataloader)
            
    def test_model_inference(self, model, dataloader, config):
        """Test model inference and output visualization."""
        sample_batch = next(iter(dataloader))
        model.eval()
        
        with torch.no_grad():
            # Forward pass
            encoder_out = model.online_encoder(sample_batch)
            sample_batch["online_embedding"] = encoder_out["online_embedding"]
            decoder_out = model.decoder(sample_batch)
            reconstructed = decoder_out["predicted_grid"]
            
            # Attribute predictions
            attribute_predictions = {}
            for attr_key in config['model']['encoder']['downstream_attributes'].keys():
                attr_head = getattr(model, f"attribute_head_{attr_key}")
                attr_out = attr_head(sample_batch)
                attribute_predictions[attr_key] = attr_out[f"predicted_{attr_key}"]
                        
            # Optional: Print results for manual inspection
            self._print_results(sample_batch, reconstructed, attribute_predictions, config)
    
    def _print_results(self, sample_batch, reconstructed, attribute_predictions, config):
        """Print reconstruction and attribute prediction results."""
        
        sample_grid_size = sample_batch["grid_size"][2].squeeze().to(torch.int32)
        original = sample_batch["padded_grid"][2].reshape(30, 30)
        prediction = reconstructed[2].reshape(30, 30)
        
        print("=== GRID RECONSTRUCTION ===")
        print("Original grid:\n", original[0:sample_grid_size[0].item(), 0:sample_grid_size[1].item()].cpu().numpy())
        print("Predicted grid:\n", np.round(prediction[0:sample_grid_size[0].item(), 0:sample_grid_size[1].item()].cpu().numpy()))
        
        print("\n=== ATTRIBUTE PREDICTIONS ===")
        for attr_key in config['model']['encoder']['downstream_attributes'].keys():
            predicted = attribute_predictions[attr_key][2].cpu().numpy()
            actual = sample_batch[attr_key][2]
            if hasattr(actual, 'cpu'):
                actual = actual.cpu().numpy()
            
            print(f"\n{attr_key.upper()}:")
            print(f"  Predicted: {predicted}")
            print(f"  Actual:    {actual}")

class TestARCTransformer:
    @pytest.fixture
    def config(self):
        """Load test configuration."""
        return load_config()
    
    @pytest.fixture
    def model(self, config):
        """Create model instance."""
        model_config = config['model']['transformer']
        shared_configs = config['model']['shared']
        
        return TransformationDescriber(
            learning_rate=model_config['learning_rate'],
            alpha=model_config['alpha'],
            **{
                "TransformationDescriber": {
                    "input_size": shared_configs['latent_size'],
                    "output_size": shared_configs['transformation_dimension_size']
                },
            }
        )
    
    @pytest.fixture
    def dataloader(self, config):
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
    
    def test_model_training(self, model, dataloader, config):
        """Test model training process."""
        trainer = L.Trainer(max_epochs=config['training']['transformer']['epochs'])
        trainer.fit(model=model, train_dataloaders=dataloader)
            
    def test_model_inference(self, model, dataloader, config):
        """Test model inference and output visualization."""
        sample_batch = next(iter(dataloader))
        model.eval()
        
        with torch.no_grad():
            # Forward pass
            pass
          

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
    test_instance = TestARCEncoder()
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