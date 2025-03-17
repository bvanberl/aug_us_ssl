import argparse
import datetime
import os
import time
import json
import gc

import pandas as pd
import yaml
import torchvision
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger
from pytorch_lightning.callbacks import LearningRateMonitor, TQDMProgressBar, ModelCheckpoint

from src.models.classifier import Classifier
from src.data.image_datasets import load_data_for_test
from src.train.utils import *

torchvision.disable_beta_transforms_warning()


if __name__ == '__main__':

    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--checkpoint_path', required=False, type=str, help='Path to checkpointed classifier')
    parser.add_argument('--image_dir', required=False, default='', type=str, help='Root directory containing images')
    parser.add_argument('--splits_dir', required=False, default='', type=str,
                        help='Root directory containing splits information')
    parser.add_argument('--nodes', default=1, type=int, help='Number of nodes')
    parser.add_argument('--gpus_per_node', default=1, type=int, help='Number of GPUs per node')
    parser.add_argument('--batch_size', required=False, type=int, help='Batch size')
    parser.add_argument('--num_test_workers', required=False, type=int, default=0, help='Number of workers for loading test set')
    parser.add_argument('--seed', required=False, type=int, help='Random seed')
    parser.add_argument('--label', required=False, type=str, help='Name of label column')
    args = vars(parser.parse_args())
    print(f"Args: {json.dumps(args, indent=2)}")

    cfg = yaml.full_load(open(os.getcwd() + "/config.yml", 'r'))
    if os.path.exists("./wandb.yml"):
        wandb_cfg = yaml.full_load(open(os.getcwd() + "/wandb.yml", 'r'))
    else:
        wandb_cfg = {'MODE': 'disabled', 'RESUME_ID': None}

    seed = args['seed'] if args['seed'] else cfg['train']['seed']
    seed_everything(seed, args["num_test_workers"] > 0)

    image_dir = args['image_dir'] if args['image_dir'] else cfg["paths"]["images"]
    splits_dir = args['splits_dir'] if args['splits_dir'] else cfg["paths"]["splits"]
    height = cfg['data']['height']
    width = cfg['data']['width']
    channels = 3
    img_dim = (channels, height, width)
    batch_size = args['batch_size']
    label_name = args['label']
    n_test_workers = args["num_test_workers"]
    num_nodes = args['nodes']
    num_gpus = args['gpus_per_node']
    convert_all_to_linear = cfg['train']['augmentation']['convert_all_to_linear']

    test_loader = load_data_for_test(
        image_dir,
        label_name,
        width,
        height,
        splits_dir,
        batch_size,
        n_test_workers=n_test_workers,
        convert_all_to_linear=convert_all_to_linear
    )

    trainer = Trainer(
        accelerator="gpu",
        devices=num_gpus,
        num_nodes=num_nodes
    )

    model = Classifier.load_from_checkpoint(args['checkpoint_path'])
    test_metrics = trainer.test(model, test_loader)[0]

    print(test_metrics)