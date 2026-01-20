import json
import os

def load_config(config_path="config.json"):
    """Memuat konfigurasi dari file JSON."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"File konfigurasi tidak ditemukan: {config_path}")
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    return config