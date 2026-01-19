import toml
import argparse
from pathlib import Path
from typing import Dict, Any

def load_config(config_name: str = None) -> Dict[str, Any]:
    """Load configuration from TOML file."""
    if config_name is None:
        config_path = Path(__file__).parent.parent.parent / "config" / "test_config.toml"
    else:
        config_path = Path(__file__).parent.parent.parent / "config" / f"{config_name}.toml"
    
    with open(config_path, 'r') as f:
        return toml.load(f)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='ARC Module Testing')
    parser.add_argument('--config', type=str, help='Path to config file')
    parser.add_argument('--epochs', type=int, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, help='Batch size')
    parser.add_argument('--sample-idx', type=int, help='Sample index for visualization')
    parser.add_argument('--mode', choices=['train', 'test', 'both'], default='both',
                       help='Run mode: train only, test only, or both')
    return parser.parse_args()