import yaml
import argparse
import os

class Config:
    """This is a Global class which holds the configuration parameters"""
    _instance = None

    def __new__(cls, config_path = None):
        if cls._instance is None:
            if config_path is None:
                raise ValueError("Config path must be provided the first time")
            cls._instance = super(Config, cls).__new__(cls)
            with open(config_path, 'r') as f:
                content = yaml.safe_load(f)
                cls._instance.__dict__.update(content)

        return cls._instance

def get_args():
    """Reads the config file name from the terminal command"""
    parser = argparse.ArgumentParser(description="GAN Training")
    parser.add_argument('--config', type=str, default='config.yaml',
                        help='Name of the config file in the config/ folder')
    args = parser.parse_args()
    config_path = os.path.join('config', args.config)
    return config_path
