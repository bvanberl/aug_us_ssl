import argparse
import datetime
import os
import time
import json

import yaml
import torchvision
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger
from pytorch_lightning.callbacks import LearningRateMonitor

from src.models.classifier import Classifier
from src.models.joint_embedding import JointEmbeddingModel
from src.models.extractors import get_extractor
from src.data.image_datasets import load_data_for_train
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
    parser.add_argument('--linear', required=False, type=int, help='0 for fine-tuning, 1 for linear')
    parser.add_argument('--extractor_weights', required=False, type=str, help='Path to saved joint embedding model')
    parser.add_argument('--image_dir', required=False, default='', type=str, help='Root directory containing images')
    parser.add_argument('--splits_dir', required=False, default='', type=str,
                        help='Root directory containing splits information')
    parser.add_argument('--nodes', default=1, type=int, help='Number of nodes')
    parser.add_argument('--gpus_per_node', default=1, type=int, help='Number of GPUs per node')
    parser.add_argument('--log_interval', default=1, type=int, help='Number of steps after which to log')
    parser.add_argument('--epochs', required=False, type=int, help='Number of epochs')
    parser.add_argument('--batch_size', required=False, type=int, help='Batch size')
    parser.add_argument('--augment_pipeline', required=False, type=str, default=None, help='Augmentation pipeline')
    parser.add_argument('--num_workers', required=False, type=int, default=0, help='Number of workers for data loading')
    parser.add_argument('--seed', required=False, type=int, help='Random seed')
    parser.add_argument('--checkpoint_path', required=False, type=str, help='Checkpoint to resume from')
    parser.add_argument('--labelled_only', required=False, type=int, help='Whether to use only examples with labels')
    args = vars(parser.parse_args())
    print(f"Args: {args}")

    num_nodes = args['nodes']
    num_gpus = args['gpus_per_node']
    world_size = num_nodes * num_gpus
    seed = args['seed'] if args['seed'] else cfg['train']['seed']
    n_workers = args["num_workers"]
    seed_everything(seed, n_workers > 0)

    # Update config with values from command-line args
    for k in cfg['data']:
        if k in args and args[k] is not None:
            cfg['data'][k] = args[k]
    for k in cfg['train']:
        if k in args and args[k] is not None:
            cfg['train'][k] = args[k]
    print(f"Config after parsing args: {cfg}")

    # Specify image directory, splits CSV directory, image shape, batch size
    image_dir = args['image_dir'] if args['image_dir'] else cfg["paths"]["images"]
    splits_dir = args['splits_dir'] if args['splits_dir'] else cfg["paths"]["splits"]
    height = cfg['data']['height']
    width = cfg['data']['width']
    channels = 3
    img_dim = (channels, height, width)
    batch_size = cfg['train']['batch_size']
    label_name = cfg['train']['label']

    # Determine data augmentation pipeline
    if args["augment_pipeline"] is not None:
        augment_pipeline = args["augment_pipeline"]
    else:
        augment_pipeline = cfg['train']['augment_pipeline']

    # Create training and validation data loaders
    train_loader, val_loader = load_data_for_train(
        image_dir,
        label_name,
        width,
        height,
        splits_dir,
        batch_size,
        augment_pipeline=augment_pipeline,
        n_workers=n_workers
    )

    # Finalize run configuration
    run_cfg = {
        'seed': seed
    }
    run_cfg.update(cfg['data'])
    run_cfg.update(cfg['train'])
    run_cfg = {k.lower(): v for k, v in run_cfg.items()}

    # Initialize wandb run
    train_type_tag = "linear" if cfg['train']['linear'] else "fine-tune"
    use_wandb = wandb_cfg["mode"] == "online"
    resume_id = wandb_cfg["resume_id"]
    if use_wandb:
        wandb_run = wandb.init(
            project=wandb_cfg['project'],
            job_type=f"train",
            entity=wandb_cfg['entity'],
            config=run_cfg,
            sync_tensorboard=False,
            tags=["classification", train_type_tag],
            id=resume_id
        )
        print(f"Run config: {wandb_run}")
        model_artifact = wandb.Artifact(f"trained_{train_type_tag}", type="model")
    else:
        wandb_run = None
        model_artifact = None

    # Define classifier for training.
    if args['checkpoint_path']:
        # Resume from checkpoint
        model = Classifier.load_from_checkpoint(args['checkpoint_path'])
        checkpoint_dir = os.path.dirname(args['checkpoint_path'])
        epochs = model.scheduler_epochs
        load_ckpt_path = args['checkpoint_path']

    else:
        # Define new model using a checkpoint
        epochs = cfg['train']['epochs']
        batches_per_epoch = len(train_loader)

        # Create feature extractor
        if cfg['train']['extractor_weights'] == 'scratch':
            extractor = get_extractor(cfg['train']['extractor'], False).cuda()
        elif cfg['train']['extractor_weights'] == 'imagenet':
            extractor = get_extractor(cfg['train']['extractor'], True).cuda()
        else:
            je_model = JointEmbeddingModel.load_from_checkpoint(cfg['train']['extractor_weights'])
            extractor = je_model.extractor

        n_classes = train_loader.dataset.n_classes
        model = Classifier(
            extractor,
            img_dim,
            n_classes,
            cfg['train']['lr_head'],
            cfg['train']['lr_extractor'],
            epochs,
            cfg['train']['weight_decay'],
            bool(cfg['train']['linear'])
        )
        #model.summary()

        # Set checkpoint/log dir and save the run config as a JSON file
        date = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        checkpoint_dir = os.path.join(cfg['paths']['model_weights'], 'classifiers', label_name, date)
        os.makedirs(checkpoint_dir, exist_ok=True)
        print(f"Checkpoint Dir: {checkpoint_dir}")
        run_cfg_path = os.path.join(checkpoint_dir, "run_cfg.json")
        with open(run_cfg_path, 'w') as f:
            json.dump(run_cfg, f)
        load_ckpt_path = None

    # Create loggers
    log_dir = os.path.join(checkpoint_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    loggers = [TensorBoardLogger(log_dir)]
    if use_wandb:
        loggers.append(WandbLogger(log_model="all"))

    # Create callbacks
    callbacks = [LearningRateMonitor(logging_interval='step')]

    # Train the model
    trainer = Trainer(
        max_epochs=epochs,
        accelerator="gpu",
        devices=num_gpus,
        num_nodes=num_nodes,
        logger=loggers,
        default_root_dir=checkpoint_dir,
        callbacks=callbacks,
        log_every_n_steps=args['log_interval']
    )
    trainer.fit(model, train_loader, val_loader, ckpt_path=load_ckpt_path)

    if use_wandb:
        wandb.finish()
