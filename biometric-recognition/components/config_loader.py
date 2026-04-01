import argparse
import configparser
import os

def get_combined_args(description="Biometric Script"):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", type=str, default="../configs/experiment1.config", help="Path to config file")
    
    # Define arguments that can be in the config OR the CLI
    parser.add_argument("--raw_dir", type=str)
    parser.add_argument("--processed_dir", type=str)
    parser.add_argument("--model_path", type=str)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--split", type=float)

    args = parser.parse_args()

    # 2. Load Config File
    config = configparser.ConfigParser()
    if os.path.exists(args.config):
        config.read(args.config)
    
    # 3. Merge Logic: CLI takes priority, then Config, then Defaults
    # Mapping: (ArgName, ConfigSection, ConfigKey, Default)
    mapping = [
        ('raw_dir', 'DATA', 'raw_dir', '/data/raw'),
        ('processed_dir', 'DATA', 'processed_dir', '/data/processed'),
        ('split', 'DATA', 'split_ratio', 0.8),
        ('epochs', 'HYPERPARAMETERS', 'epochs', 10),
        ('batch_size', 'HYPERPARAMETERS', 'batch_size', 32),
        ('lr', 'HYPERPARAMETERS', 'lr', 0.001),
        ('model_path', 'MODEL', 'model_path', '/data/model/model.pth'),
    ]

    for arg_name, section, key, default in mapping:
        val = getattr(args, arg_name)
        if val is None: # If not provided in CLI
            if config.has_option(section, key):
                # Handle type conversion from string config
                conf_val = config.get(section, key)
                if isinstance(default, int): conf_val = int(conf_val)
                elif isinstance(default, float): conf_val = float(conf_val)
                setattr(args, arg_name, conf_val)
            else:
                setattr(args, arg_name, default)
    
    return args
