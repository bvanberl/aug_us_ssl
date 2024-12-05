from typing import List, Tuple, Optional, Dict
from abc import abstractmethod

import torch
from torch.nn import Module
from torch import Tensor
import pytorch_lightning as pl
import torchsummary

from src.models.extractors import get_extractor
from src.models.projectors import get_projector
from src.custom.optimizers import LARS
from src.custom.schedulers import WarmupCosineDecayLR
from src.losses.losses import *

class JointEmbeddingModel(pl.LightningModule):

    def __init__(
            self,
            method: str,
            hparams: Dict,
            input_shape: Tuple[int, int, int],
            extractor_name: str,
            imagenet_weights: bool,
            projector_nodes: List[int],
            batch_size: int,
            batches_per_epoch: int,
            max_lr: float,
            extractor_cutoff_layers: int = 0,
            projector_bias: bool = False,
            weight_decay: float = 1e-6,
            warmup_epochs: int = 10,
            scheduler_epochs: int = 100,
            world_size=1,
            train_metric_freq = 100
        ):
        """
        Args:
            method: Name of self-supervised learning method
            input_shape: Expected shape of input images (C, H, W)
            extractor_name: Feature extractor identifier
            imagenet_weights: If True, initializes extractor
                with ImageNet-pretrained weights
            projector_nodes: Number of nodes in each fully connected layer
            batch_size: Batch size
            batches_per_epoch: Number of batches per epoch
            extractor_cutoff_layers: Number of layers to remove from the end of the
                extractor model.
            projector_bias: If True, use biases in fully connected layers
            weight_decay: Weight decay for optimizer
            warmup_epochs: Number of warmup epochs for cosine scheduler
            scheduler_epochs: Total number of epochs for cosine scheduler
        """

        assert warmup_epochs <= scheduler_epochs, \
            "Number of warmup epochs cannot exceed scheduler epochs"
        super().__init__()
        self.save_hyperparameters()
        self.distributed = world_size > 1

        # Define the self-supervised loss
        if method.lower() == 'simclr':
            self.loss = SimCLRLoss(tau=hparams["tau"], distributed=self.distributed)
        elif method.lower() == 'barlow_twins':
            self.loss = BarlowTwinsLoss(batch_size, lambda_=hparams["lambda_"], distributed=self.distributed)
        elif method.lower() == 'vicreg':
            self.loss = VICRegLoss(batch_size, lambda_=hparams["lambda_"], mu=hparams["mu"], nu=hparams["nu"],
                                 distributed=self.distributed)
        else:
            raise NotImplementedError(f'{method} is not currently supported.')

        self.input_shape = input_shape
        self.batch_size = batch_size
        self.batches_per_epoch = batches_per_epoch
        self.max_lr = max_lr
        self.weight_decay = weight_decay
        self.warmup_epochs = warmup_epochs
        self.scheduler_epochs = scheduler_epochs
        self.train_metric_freq = train_metric_freq

        self.extractor = get_extractor(
            extractor_name,
            imagenet_weights,
            extractor_cutoff_layers
        )

        self.h_dim = self.extractor(torch.randn(*((1,) + input_shape))).shape[-1]
        self.projector = get_projector(
            self.h_dim,
            projector_nodes,
            use_bias=projector_bias
        )

    def forward(self, x: Tensor) -> Tensor:

        # Compute feature representations
        h = self.extractor(x)

        # Compute embeddings
        z = self.projector(h)

        return z

    def training_step(self, batch, batch_idx):
        x0, x1 = batch

        # Get embeddings for each view
        z0 = self.forward(x0)
        z1 = self.forward(x1)

        # Compute self-supervised loss
        loss, loggables = self.loss(z0, z1)

        # Log the loss, loss components, standard deviation of embeddings
        if batch_idx % self.train_metric_freq == 0:
            self.log('train/loss', loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=self.distributed)
            self.log_dict({f"train/{key}": loggables[key] for key in loggables}, prog_bar=True, on_step=True, on_epoch=True, sync_dist=self.distributed)
            self.log(f"train/z0_std", z0.std(dim=1).mean(), on_step=True, on_epoch=True, sync_dist=self.distributed)
            self.log(f"train/z1_std", z1.std(dim=1).mean(), on_step=True, on_epoch=True, sync_dist=self.distributed)

        return loss

    def validation_step(self, batch, batch_idx):
        x0, x1 = batch

        # Get embeddings for each view
        z0 = self.forward(x0)
        z1 = self.forward(x1)

        # Compute self-supervised loss
        loss, loggables = self.loss(z0, z1)

        # Log the loss, loss components, standard deviation of embeddings
        self.log(f"val/loss", loss, sync_dist=self.distributed, prog_bar=True, on_step=True, on_epoch=True)
        self.log_dict({f"val/{key}": loggables[key] for key in loggables}, sync_dist=self.distributed, prog_bar=True, on_step=True, on_epoch=True)
        self.log(f"val/z0_std", z0.std(dim=1).mean(), sync_dist=self.distributed, on_step=True, on_epoch=True)
        self.log(f"val/z1_std", z1.std(dim=1).mean(), sync_dist=self.distributed, on_step=True, on_epoch=True)

        return loss

    def configure_optimizers(self):
        optimizer = LARS(self.parameters(), lr=0, weight_decay=self.weight_decay)
        scheduler = WarmupCosineDecayLR(
            optimizer,
            self.warmup_epochs,
            self.scheduler_epochs,
            self.batches_per_epoch,
            self.max_lr,
            self.batch_size
        )
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]

    def summary(self):
        torchsummary.summary(self.extractor.cuda(), input_size=self.input_shape)
        torchsummary.summary(self.projector.cuda(), input_size=(self.h_dim,))
