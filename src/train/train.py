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
from src.models.ssd import SSDLite
from src.models.joint_embedding import JointEmbeddingModel
from src.models.extractors import get_extractor
from src.data.image_datasets import load_data_for_train, k_way_train_split
from src.train.utils import *

torchvision.disable_beta_transforms_warning()

def train(
        cfg: dict,
        args: dict,
        train_clips: pd.DataFrame = None,
        test_clips: pd.DataFrame = None,
        ckpt_metric: Optional[str] = None,
        perform_test: bool = False,
        save_checkpoints: bool = True,
):
    num_nodes = args['nodes']
    num_gpus = args['gpus_per_node']
    world_size = num_nodes * num_gpus
    seed = args['seed'] if args['seed'] else cfg['train']['seed']
    n_train_workers = args["num_train_workers"]
    n_val_workers = args["num_val_workers"]
    n_test_workers = args["num_test_workers"]

    # Specify image directory, splits CSV directory, image shape, batch size
    image_dir = args['image_dir'] if args['image_dir'] else cfg["paths"]["images"]
    mask_dir = args['mask_dir'] if args['mask_dir'] else cfg["paths"]["masks"]
    splits_dir = args['splits_dir'] if args['splits_dir'] else cfg["paths"]["splits"]
    square_roi = args['square_roi'] if args['square_roi'] else bool(cfg['pretrain']['square_roi'])
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
    min_crop = args['min_crop']
    convert_all_to_linear = cfg['train']['augmentation']['convert_all_to_linear']

    # Create training and validation data loaders
    train_loader, val_loader, test_loader = load_data_for_train(
        image_dir,
        label_name,
        width,
        height,
        splits_dir,
        batch_size,
        augment_pipeline=augment_pipeline,
        n_train_workers=n_train_workers,
        n_val_workers=n_val_workers,
        n_test_workers=n_test_workers,
        train_clips=train_clips,
        k_fold_test_clips=test_clips,
        min_crop=min_crop,
        mask_dir=mask_dir,
        square_roi=square_roi,
        convert_all_to_linear=convert_all_to_linear
    )
    if ckpt_metric is None:
        val_loader = None   # Don't get validation set metrics if not monitoring val performance

    # Finalize run configuration
    run_cfg = {
        'seed': seed
    }
    run_cfg.update(cfg['data'])
    run_cfg.update(cfg['train'])
    run_cfg = {k.lower(): v for k, v in run_cfg.items()}
    print(run_cfg)

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
        if args['epochs'] is not None:
            epochs = args['epochs']
        else:
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
        if label_name == 'pl_label':
            model = SSDLite(
                extractor,
                img_dim,
                n_classes,
                cfg['train']['lr_head'],
                cfg['train']['lr_extractor'],
                epochs,
                cfg['train']['weight_decay'],
                bool(cfg['train']['linear']),
                world_size,
                min_ratio=cfg['train']['obj_det']['min_ratio'],
                max_ratio=cfg['train']['obj_det']['max_ratio'],
                aspect_ratios=cfg['train']['obj_det']['aspect_ratios']
            )
        else:
            model = Classifier(
                extractor,
                img_dim,
                n_classes,
                cfg['train']['lr_head'],
                cfg['train']['lr_extractor'],
                epochs,
                cfg['train']['weight_decay'],
                bool(cfg['train']['linear']),
                world_size
            )
        model.summary()
        print("\n\n\n")

        # Set checkpoint/log dir and save the run config as a JSON file
        date = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        if args['checkpoint_dir']:
            checkpoint_dir = args['checkpoint_dir']
        else:
            checkpoint_dir = os.path.join(cfg['paths']['model_weights'], 'classifiers', label_name, date)
        os.makedirs(checkpoint_dir, exist_ok=True)
        print(f"Checkpoint Dir: {checkpoint_dir}")
        run_cfg_path = os.path.join(checkpoint_dir, "run_cfg.json")
        with open(run_cfg_path, 'w') as f:
            json.dump(run_cfg, f, indent=4)
        load_ckpt_path = None

    # Create loggers
    log_dir = os.path.join(checkpoint_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    loggers = [TensorBoardLogger(log_dir)]
    if use_wandb:
        loggers.append(WandbLogger(log_model="all"))

    # Create callbacks
    if ckpt_metric is not None and 'loss' in ckpt_metric:
        mode = 'min'
    else:
        mode = 'max'

    callbacks = [
        LearningRateMonitor(logging_interval='step'),
        TQDMProgressBar(refresh_rate=args['log_interval'])
    ]
    if save_checkpoints:
        ckpt_callback = ModelCheckpoint(monitor=ckpt_metric, dirpath=checkpoint_dir, filename='best-{epoch}-{step}',
                                mode=mode)
        callbacks.append(ckpt_callback)

    # Train the model
    trainer = Trainer(
        max_epochs=epochs,
        accelerator="gpu",
        devices=num_gpus,
        num_nodes=num_nodes,
        logger=loggers,
        default_root_dir=checkpoint_dir,
        callbacks=callbacks,
        log_every_n_steps=args['log_interval'],
        deterministic=args['deterministic'],
        enable_checkpointing=save_checkpoints
    )
    trainer.fit(model, train_loader, val_loader, ckpt_path=load_ckpt_path)

    
    test_metrics = {}
    if perform_test:
        if ckpt_metric is not None and save_checkpoints:
            # Restore the model with lowest validation set loss and evaluate it on the test set
            model_path = ckpt_callback.best_model_path
            print(f"Best model saved at: {model_path}")
            if label_name == 'pl_label':
                model = SSDLite.load_from_checkpoint(model_path)
            else:
                model = Classifier.load_from_checkpoint(model_path)
        test_metrics = trainer.test(model, test_loader)[0]

    if use_wandb:
        wandb.finish()

    return test_metrics


def label_efficiency_experiment(cfg: dict, args: dict):
    """
    Splits training set into N chunks by patient. Trains a model on each
    subset and records test metrics.
    :param cfg: The config.yaml file dictionary
    :param args: Command-line arguments
    """

    n_splits = cfg['train']['n_splits_label_eff']

    train_dfs = k_way_train_split(
        args['splits_dir'],
        n_splits,
        args['label'],
        seed=args['seed'],
        stratify_by_label=True,
        group_col='patient_id'
    )

    base_checkpoint_dir = args['checkpoint_dir']
    metrics_df = pd.DataFrame()

    for i in range(n_splits):
        print(f"Trial {i + 1} / {n_splits}.\n\n")

        args['checkpoint_dir'] = os.path.join(base_checkpoint_dir, f"split{i}")
        test_metrics = train(cfg, args, train_dfs[i], save_checkpoints=True, perform_test=True)

        if i == 0:
            metrics_df = pd.DataFrame([test_metrics])
        else:
            new_row = pd.DataFrame([test_metrics])
            metrics_df = pd.concat([metrics_df, new_row], ignore_index=True)
        gc.collect()
        torch.cuda.empty_cache()

    metrics_df.to_csv(os.path.join(base_checkpoint_dir, "label_efficiency_results.csv"), index=False)


def k_fold_cross_validation(cfg: dict, args: dict):
    """
    Splits training set by patient into k folds. Performs k-fold cross-
    validation and records test metrics for each trial.
    :param cfg: The config.yaml file dictionary
    :param args: Command-line arguments
    """

    k = cfg['train']['k_folds']

    train_dfs = k_way_train_split(
        args['splits_dir'],
        k,
        args['label'],
        seed=args['seed'],
        stratify_by_label=True,
        group_col='patient_id'
    )

    base_checkpoint_dir = args['checkpoint_dir']
    metrics_df = pd.DataFrame()

    for i in range(k):
        print(f"k-fold Cross-Validation Trial {i + 1} / {k}.\n\n")

        args['checkpoint_dir'] = os.path.join(base_checkpoint_dir, f"split{i}")

        train_idxs = [j for j in list(range(k)) if j != i]
        train_df = pd.concat([train_dfs[j] for j in train_idxs])
        cur_fold_df = train_dfs[i]
        test_metrics = train(cfg, args, train_clips=train_df, test_clips=cur_fold_df, perform_test=True, save_checkpoints=False)

        if i == 0:
            metrics_df = pd.DataFrame([test_metrics])
        else:
            new_row = pd.DataFrame([test_metrics])
            metrics_df = pd.concat([metrics_df, new_row], ignore_index=True)
        gc.collect()
        torch.cuda.empty_cache()

    metrics_df.to_csv(os.path.join(base_checkpoint_dir, "k_fold_cv_results.csv"), index=False)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--linear', required=False, type=int, help='0 for fine-tuning, 1 for linear')
    parser.add_argument('--extractor_weights', required=False, type=str, help='Path to saved joint embedding model')
    parser.add_argument('--image_dir', required=False, default='', type=str, help='Root directory containing images')
    parser.add_argument('--mask_dir', required=False, default='', type=str, help='Root directory containing masks')
    parser.add_argument('--splits_dir', required=False, default='', type=str,
                        help='Root directory containing splits information')
    parser.add_argument('--nodes', default=1, type=int, help='Number of nodes')
    parser.add_argument('--gpus_per_node', default=1, type=int, help='Number of GPUs per node')
    parser.add_argument('--log_interval', default=1, type=int, help='Number of steps after which to log')
    parser.add_argument('--epochs', required=False, type=int, help='Number of epochs')
    parser.add_argument('--batch_size', required=False, type=int, help='Batch size')
    parser.add_argument('--augment_pipeline', required=False, type=str, default=None, help='Augmentation pipeline')
    parser.add_argument('--num_train_workers', required=False, type=int, default=0, help='Number of workers for loading train set')
    parser.add_argument('--num_val_workers', required=False, type=int, default=0, help='Number of workers for loading val set')
    parser.add_argument('--num_test_workers', required=False, type=int, default=0, help='Number of workers for loading test set')
    parser.add_argument('--seed', required=False, type=int, help='Random seed')
    parser.add_argument('--checkpoint_dir', required=False, type=str, help='Directory in which to save checkpoints')
    parser.add_argument('--checkpoint_path', required=False, type=str, help='Checkpoint to resume from')
    parser.add_argument('--labelled_only', required=False, type=int, help='Whether to use only examples with labels')
    parser.add_argument('--label', required=False, type=str, help='Name of label column')
    parser.add_argument('--deterministic', action='store_true', help='If provided, sets the `deterministic` flag in Trainer')
    parser.add_argument('--test', action='store_true', help='If provided, performs test set evaluation')
    parser.add_argument('--experiment_type', type=str, default='single_train', required=False, help='Type of training experiment')
    parser.add_argument('--k-folds', type=int, default=10, required=False, help='Number of folds for k-fold cross-validation')
    parser.add_argument('--min_crop', required=False, type=float, default=0.7, help='Minimum crop for random crop & resize')
    parser.add_argument('--square_roi', required=False, type=int, help='1 if images cropped and resized to square; 0 otherwise')

    args = vars(parser.parse_args())
    print(f"Args: {json.dumps(args, indent=2)}")

    cfg = yaml.full_load(open(os.getcwd() + "/config.yml", 'r'))
    if os.path.exists("./wandb.yml"):
        wandb_cfg = yaml.full_load(open(os.getcwd() + "/wandb.yml", 'r'))
    else:
        wandb_cfg = {'MODE': 'disabled', 'RESUME_ID': None}

    seed = args['seed'] if args['seed'] else cfg['train']['seed']
    seed_everything(seed, args["num_train_workers"] > 0)

    # Update config with values from command-line args
    for k in cfg['data']:
        if k in args and args[k] is not None:
            cfg['data'][k] = args[k]
    for k in cfg['train']:
        if k in args and args[k] is not None:
            cfg['train'][k] = args[k]
    print(f"Data config after parsing args:\n {json.dumps(cfg['data'], indent=2)}")
    print(f"Train config after parsing args:\n {json.dumps(cfg['train'], indent=2)}")

    if args['experiment_type'] == 'single_train':
        test_metrics = train(cfg, args, ckpt_metric='val/loss', save_checkpoints=True, perform_test=args['test'])
        print(f"Test metrics: {json.dumps(test_metrics, indent=2)}")
    elif args['experiment_type'] == 'cross_validation':
        k_fold_cross_validation(cfg, args)
    elif args['experiment_type'] == 'label_efficiency':
        label_efficiency_experiment(cfg, args)
    else:
        raise ValueError(f"Unknown experiment type: {args['experiment_type']}")