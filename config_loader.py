"""Configuration loader for data paths.

This module loads data paths from config.yaml and provides
utility functions to access them throughout the codebase.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Any
import yaml


def load_config(config_path: str | Path | None = None) -> Dict[str, Any]:
    """Load configuration from YAML file.
    
    Parameters
    ----------
    config_path : str | Path | None
        Path to config.yaml. If None, looks for config.yaml in the repo root.
        
    Returns
    -------
    Dict[str, Any]
        Configuration dictionary
    """
    if config_path is None:
        repo_root = Path(__file__).parent
        config_path = repo_root / "config.yaml"
    else:
        config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(
            f"Configuration file not found: {config_path}\n"
            "Please create config.yaml with your data paths. "
            "See config.yaml.example for template."
        )
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Expand environment variables and ${var} references
    config = _expand_vars(config)
    
    return config


def _expand_vars(config: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively expand ${var} references and environment variables."""
    if isinstance(config, dict):
        result = {}
        for key, value in config.items():
            result[key] = _expand_vars(value)
        
        # Second pass to resolve ${key} references
        for key, value in result.items():
            if isinstance(value, str) and '${' in value:
                result[key] = _resolve_references(value, result)
        
        return result
    elif isinstance(config, list):
        return [_expand_vars(item) for item in config]
    elif isinstance(config, str):
        return os.path.expandvars(config)
    else:
        return config


def _resolve_references(text: str, config: Dict[str, Any]) -> str:
    """Resolve ${key} references in text."""
    import re
    pattern = r'\$\{([^}]+)\}'
    
    def replacer(match):
        key = match.group(1)
        value = config.get(key, match.group(0))  # Keep ${key} if not found
        return str(value) if value is not None else match.group(0)
    
    return re.sub(pattern, replacer, text)


def get_data_paths(config: Dict[str, Any] | None = None) -> Dict[str, str]:
    """Get data paths from configuration.
    
    Parameters
    ----------
    config : Dict[str, Any] | None
        Configuration dictionary. If None, loads from config.yaml
        
    Returns
    -------
    Dict[str, str]
        Dictionary with data paths:
        - eyes_closed_data
        - utah_coordinates  
        - channel_area_mapping
        - deleted_electrodes
    """
    if config is None:
        config = load_config()
    
    return {
        'eyes_closed_data': config['eyes_closed_data'],
        'utah_coordinates': config['utah_coordinates'],
        'channel_area_mapping': config['channel_area_mapping'],
        'deleted_electrodes': config['deleted_electrodes'],
    }


def get_monkey_config(monkey: str, config: Dict[str, Any] | None = None) -> Dict[str, Any]:
    """Get monkey-specific configuration.
    
    Parameters
    ----------
    monkey : str
        Monkey identifier (monkey_L or monkey_A)
    config : Dict[str, Any] | None
        Configuration dictionary. If None, loads from config.yaml
        
    Returns
    -------
    Dict[str, Any]
        Monkey configuration with keys:
        - full_name
        - utah_array_file
        - pixels_per_mm
        - utah_max
    """
    if config is None:
        config = load_config()
    
    monkeys = config.get('monkeys', {})
    if monkey not in monkeys:
        raise ValueError(
            f"Unknown monkey '{monkey}'. "
            f"Available monkeys: {list(monkeys.keys())}"
        )
    
    return monkeys[monkey]


if __name__ == "__main__":
    # Test configuration loading
    try:
        config = load_config()
        print("Configuration loaded successfully!")
        print(f"\nData root: {config['data_root']}")
        print(f"\nData paths:")
        for key, value in get_data_paths(config).items():
            print(f"  {key}: {value}")
        print(f"\nMonkeys configured: {list(config['monkeys'].keys())}")
    except Exception as e:
        print(f"Error loading configuration: {e}")
