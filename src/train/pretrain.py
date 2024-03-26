import argparse
import datetime
import os
import time
import json

import yaml
import torch.distributed as dist
import torchsummary
import torchvision
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger

from src.models.joint_embedding import JointEmbeddingModel
from src.data.image_datasets import load_data_for_pretrain
from src.train.utils import *

torchvision.disable_beta_transforms_warning()
torch.set_float32_matmul_precision('high')

cfg = yaml.full_load(open(os.getcwd() + "/config.yml", 'r'))
if os.path.exists("./wandb.yml"):
    wandb_cfg = yaml.full_load(open(os.getcwd() + "/wandb.yml", 'r'))
else:
    wandb_cfg = {'MODE': 'disabled', 'RESUME_ID': None}

if __name__ == '__main__':

    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--method', required=False, type=str, help='SSL Pre-training method')
    parser.add_argument('--image_dir', required=False, default='', type=str, help='Root directory containing images')
    parser.add_argument('--mask_dir', required=False, default='', type=str, help='Root directory containing masks')
    parser.add_argument('--splits_dir', required=False, default='', type=str,
                        help='Root directory containing splits information')
    parser.add_argument('--nodes', default=1, type=int, help='Number of nodes')
    parser.add_argument('--gpus_per_node', default=1, type=int, help='Number of GPUs per node')
    parser.add_argument('--dist_url', default="localhost", type=str, help='URL used to set up distributed training')
    parser.add_argument('--dist_backend', default='gloo', type=str, help='Backend for distributed package')
    parser.add_argument('--log_interval', default=1, type=int, help='Number of steps after which to log')
    parser.add_argument('--epochs', required=False, type=int, help='Number of pretraining epochs')
    parser.add_argument('--batch_size', required=False, type=int, help='Pretraining batch size')
    parser.add_argument('--max_lr', required=False, type=float, help='Base learning rate')
    parser.add_argument('--lambda_', required=False, default=None, type=float, help='Invariance term weight')
    parser.add_argument('--augment_pipeline', required=False, type=str, default=None, help='Augmentation pipeline')
    parser.add_argument('--num_workers', required=False, type=int, default=0, help='Number of workers for data loading')
    parser.add_argument('--seed', required=False, type=int, help='Random seed')
    parser.add_argument('--checkpoint_name', required=False, type=str, help='Augmentation pipeline')
    args = vars(parser.parse_args())
    print(f"Args: {args}")

    num_nodes = args['nodes']
    num_gpus = args['gpus_per_node']
    world_size = num_nodes * num_gpus
    seed = args['seed'] if args['seed'] else cfg['pretrain']['seed']
    n_workers = args["num_workers"]
    seed_everything(seed, n_workers > 0)

    # Update config with values from command-line args
    for k in cfg['data']:
        if k in args and args[k] is not None:
            cfg['data'][k] = args[k]
    for k in cfg['pretrain']:
        if k in args and args[k] is not None:
            cfg['pretrain'][k] = args[k]
    print(f"Config after parsing args: {cfg}")

    # Determine SSL method and any associated hyperparameters
    method = cfg['pretrain']['method'].lower()
    assert method in ['simclr', 'barlow_twins', 'vicreg'], \
        f"Unsupported pretraining method: {method}"
    hparams = {k.lower(): v for k, v in cfg['pretrain']['hparams'].pop(method.lower()).items()}
    for k in args:
        if k in hparams and args[k] is not None:
            hparams[k] = args[k]

    # Specify image directory, splits CSV directory, image shape, batch size
    image_dir = args['image_dir'] if args['image_dir'] else cfg["paths"]["images"]
    mask_dir = args['mask_dir'] if args['mask_dir'] else cfg["paths"]["masks"]
    splits_dir = args['splits_dir'] if args['splits_dir'] else cfg["paths"]["splits"]
    height = cfg['data']['height']
    width = cfg['data']['width']
    channels = 3
    img_dim = (channels, height, width)
    batch_size = cfg['pretrain']['batch_size']

    # Determine data augmentation pipeline
    if args["augment_pipeline"] is not None:
        augment_pipeline = args["augment_pipeline"]
    else:
        augment_pipeline = cfg['pretrain']['augment_pipeline']
    # if augment_pipeline == 'august':
    #     aug_params = cfg['augment'][augment_pipeline]
    #     for k in aug_params:
    #         if k in args and args[k] is not None:
    #             aug_params[k] = args[k]
    #     hparams['augmentation'] = aug_params
    # else:
    #     hparams['augmentation'] = {}
    print(f"Method hyperparameters: {hparams}")

    # Create training and validation data loaders
    train_loader, val_loader = load_data_for_pretrain(
        image_dir,
        mask_dir,
        width,
        height,
        splits_dir,
        batch_size,
        augment_pipeline=augment_pipeline,
        use_unlabelled=cfg['pretrain']['use_unlabelled'],
        n_workers=n_workers,
        **hparams
    )

    # Finalize run configuration
    run_cfg = {
        'seed': seed
    }
    run_cfg.update(cfg['data'])
    run_cfg.update(cfg['pretrain'])
    run_cfg = {k.lower(): v for k, v in run_cfg.items()}
    run_cfg.update(hparams)

    # Initialize wandb run
    use_wandb = wandb_cfg["mode"] == "online"
    resume_id = wandb_cfg["resume_id"]
    if use_wandb:
        wandb_run = wandb.init(
            project=wandb_cfg['project'],
            job_type=f"pretrain",
            entity=wandb_cfg['entity'],
            config=run_cfg,
            sync_tensorboard=False,
            tags=["pretrain", method],
            id=resume_id
        )
        print(f"Run config: {wandb_run}")
        model_artifact = wandb.Artifact(f"pretrained_{method}", type="model")
    else:
        wandb_run = None
        model_artifact = None

    # Define model for pretraining. Includes feature extractor and projector.
    epochs = cfg['pretrain']['epochs']
    batches_per_epoch = len(train_loader)
    model = JointEmbeddingModel(
        method,
        hparams,
        img_dim,
        cfg['pretrain']['extractor'],
        cfg['pretrain']['imagenet_weights'],
        cfg['pretrain']['proj_nodes'],
        batch_size,
        batches_per_epoch,
        cfg['pretrain']['max_lr'],
        extractor_cutoff_layers=cfg['pretrain']['n_cutoff_layers'],
        projector_bias=cfg['pretrain']['use_bias'],
        weight_decay=cfg['pretrain']['weight_decay'],
        warmup_epochs=cfg['pretrain']['warmup_epochs'],
        scheduler_epochs=epochs,
        world_size=world_size
    )
    #model.summary()

    # Set checkpoint/log dir and save the run config as a JSON file
    if args['checkpoint_name']:
        checkpoint_name = args['checkpoint_name']
    else:
        checkpoint_name = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    checkpoint_dir = os.path.join(cfg['paths']['model_weights'], 'pretrained', method, checkpoint_name)
    os.makedirs(checkpoint_dir, exist_ok=True)
    print(f"Checkpoint Dir: {checkpoint_dir}")
    run_cfg_path = os.path.join(checkpoint_dir, "run_cfg.json")
    with open(run_cfg_path, 'w') as f:
        json.dump(run_cfg, f)

    # Create loggers
    log_dir = os.path.join(checkpoint_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    loggers = [TensorBoardLogger(log_dir)]
    if use_wandb:
        loggers.append(WandbLogger(log_model="all"))

    # Train the model
    trainer = Trainer(
        max_epochs=epochs,
        accelerator="gpu",
        devices=num_gpus,
        num_nodes=num_nodes,
        logger=loggers,
        default_root_dir=checkpoint_dir
    )
    trainer.fit(model, train_loader, val_loader)

    if use_wandb:
        wandb.finish()
