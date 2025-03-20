import argparse
import os

from src import config
from src.ESLAM import ESLAM
import wandb

def main():
    parser = argparse.ArgumentParser(
        description='Arguments for running ESLAM.'
    )
    parser.add_argument('config', type=str, help='Path to config file.')
    parser.add_argument('--input_folder', type=str,
                        help='input folder, this have higher priority, can overwrite the one in config file')
    parser.add_argument('--output', type=str,
                        help='output folder, this have higher priority, can overwrite the one in config file')
    args = parser.parse_args()

    cfg = config.load_config(args.config, 'configs/ESLAM.yaml')
    wandb_run = wandb.init(
        # Set the project where this run will be logged
        project=cfg['project_name'],
        # Track hyperparameters and run metadata
        config={
            "algo": "ActiveNGF"
        },
        name="ActiveNGF"
    )
    eslam = ESLAM(cfg, args, wandb_run)
    eslam.run()

if __name__ == '__main__':
    os.environ['WANDB_DISABLED'] = "true"
    main()
