import toml
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