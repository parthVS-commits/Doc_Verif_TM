"""
Initialization file for the config package

This file ensures that the configuration is loaded and validated 
when the config package is imported.
"""

from .settings import Config

# Validate configuration on import
Config.validate_config()