#!/usr/bin/env python3
"""
YAML configuration loader for pose estimation services.
"""
import yaml
import os
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

class ConfigLoader:
    """Load and parse YAML configuration files for pose estimation services."""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path
        self.config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        if self.config_path is None:
            logger.info("No configuration file path provided. Using Environment Variables.")
            return {}
        """Load YAML configuration file."""
        try:
            # Try different possible paths
            possible_paths = [
                self.config_path,
                os.path.join("../", self.config_path),
                os.path.join("../../", self.config_path),
                os.path.join("/app/config", self.config_path),
                os.path.join("/config", self.config_path)
            ]
            
            for path in possible_paths:
                if os.path.exists(path):
                    logger.info(f"Loading config from: {path}")
                    with open(path, 'r', encoding='utf-8') as f:
                        return yaml.safe_load(f)
            
            logger.warning(f"Config file not found in any of the paths: {possible_paths}")
            return {}
            
        except Exception as e:
            logger.error(f"Error loading config file: {e}")
            return {}
    
    def get_service_config(self, service_name: str) -> Dict[str, Any]:
        if self.config_path is None:
            logger.info("No configuration file path provided. Using Environment Variables.")
            return {}

        """Get configuration for a specific service."""
        service_config = self.config.get(service_name, {})
        global_config = self.config.get('global', {})
        
        # Merge global and service-specific configs
        merged_config = {**global_config, **service_config}
        
        return merged_config
    
    def get_input_config_string(self, service_name: str, input_key: str = 'input') -> Optional[str]:
        """Get input configuration string for a service."""
        service_config = self.get_service_config(service_name)
        input_config = service_config.get(input_key, {})
        
        if isinstance(input_config, dict):
            if 'config' in input_config:
                return input_config['config']
            else:
                # Build config string from dict
                return self._build_config_string(input_config)
        elif isinstance(input_config, str):
            return input_config
        
        return None
    
    def get_output_config_string(self, service_name: str) -> Optional[str]:
        """Get output configuration string for a service."""
        service_config = self.get_service_config(service_name)
        output_config = service_config.get('output', {})
        
        if isinstance(output_config, dict):
            if 'config' in output_config:
                return output_config['config']
            else:
                return self._build_config_string(output_config)
        elif isinstance(output_config, str):
            return output_config
        
        return None
    
    def get_multi_input_config_strings(self, service_name: str) -> Dict[str, str]:
        """Get multiple input configuration strings for services with multiple inputs."""
        service_config = self.get_service_config(service_name)
        input_config = service_config.get('input', {})
        
        result = {}
        for key, config in input_config.items():
            if isinstance(config, dict) and 'config' in config:
                result[key] = config['config']
        
        return result
    
    def _build_config_string(self, config_dict: Dict[str, Any]) -> str:
        """Build configuration string from dictionary."""
        parts = []
        for key, value in config_dict.items():
            parts.append(f"{key}={value}")
        return ",".join(parts)
    
    def get_devices(self, service_name: str = None) -> str:
        """Get devices configuration."""
        if service_name:
            service_config = self.get_service_config(service_name)
            return service_config.get('devices', 'cpu')
        else:
            return self.config.get('global', {}).get('devices', 'cpu')
    
    def get_log_level(self) -> str:
        """Get log level."""
        return self.config.get('global', {}).get('log_level', 'INFO')
    
    def get_backend(self, service_name: str = None) -> str:
        """Get backend configuration."""
        if service_name:
            service_config = self.get_service_config(service_name)
            return service_config.get('backend', 'onnxruntime')
        else:
            return self.config.get('global', {}).get('backend', 'onnxruntime') 