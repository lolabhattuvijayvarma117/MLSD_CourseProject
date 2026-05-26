import os
import yaml

def load_config(config_path=None):
    """
    Load configurations from a YAML file.
    """
    if config_path is None:
        # Resolve path relative to current file's directory
        root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        # Prefer config/config.yaml, fallback to params.yaml at root
        config_path = os.path.join(root_dir, "config", "config.yaml")
        if not os.path.exists(config_path):
            config_path = os.path.join(root_dir, "params.yaml")
            
    if not os.path.exists(config_path):
        return {}
        
    with open(config_path, "r") as f:
        try:
            return yaml.safe_load(f)
        except yaml.YAMLError:
            return {}
