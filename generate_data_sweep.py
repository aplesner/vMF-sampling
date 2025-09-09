
import argparse
import logging

from src.config import VMFConfig
from src.logger import setup_logging
import generate_data


if __name__ == "__main__":
    # Load configuration and set up logging
    parser = argparse.ArgumentParser(
        description='Generate samples from vMF distribution based on config file'
    )
    parser.add_argument('--config', 
                        required=False,
                        help='Path to configuration file (JSON or YAML) to override defaults')
    
    args = parser.parse_args()

    # Load base configuration at `generation_configs/base.yaml`
    base_config = VMFConfig.from_yaml('generation_configs/base.yaml')

    # Override base configuration with any user-specified values
    if args.config:
        user_config = VMFConfig.from_yaml(args.config) if args.config.endswith(('.yaml', '.yml')) else VMFConfig.from_json(args.config)
        config = base_config.model_copy(update=user_config.model_dump(), deep=True)
    else:
        config = base_config

    setup_logging(config.verbosity)

    logging.info("Starting vMF sampling...")

    generate_data.main(config)