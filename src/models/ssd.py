from typing import Tuple, Dict
import gc

import torch
import torch.nn as nn
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch import Tensor
import pytorch_lightning as pl
import torchmetrics as tm
import torchsummary
from torchvision.models.detection import SSD

class SSDLite(pl.LightningModule):
    
    def __init__(
            self,
            extractor: nn.Module,
            input_shape: Tuple[int, int, int],
            num_classes: int,
            lr_head: float,
            lr_extractor: float,
            epochs: int = 10,
            weight_decay: float = 1e-6,
            frozen_backbone: bool = False,
            world_size: int = 1,
            train_metric_freq: int = 100
        ):
        """
        Args:
            extractor: Feature extractor
            input_shape: Expected shape of input images (C, H, W)
            lr_head: Learning rate for classification head (final layer)
            lr_extractor: Learning rate for feature extractor
            epochs: Number of epochs
            weight_decay: Weight decay for optimizer
        """

        super().__init__()
        self.save_hyperparameters()

        extractor = extractor
        self.frozen_backbone = frozen_backbone
        if self.frozen_backbone:
            extractor.requires_grad_(False)
            extractor.eval()
        self.h_dim = self.extractor(torch.randn(*((1,) + input_shape)).cuda()).shape[-1]
            
        aspect_ratios = [[2, 3]] * 6
        scales = [0.1, 0.2, 0.4, 0.6, 0.8, 1.0]
        anchor_generator = DefaultBoxGenerator(aspect_ratios, scales=scales)
        classification_head = SSDLiteClassificationHead(
        in_channels=[96, self.hdim], num_anchors=6, num_classes=num_classes
        )
        regression_head = SSDLiteRegressionHead(
            in_channels=[96, self.hdim], num_anchors=6
        )
        
        head = SSDLiteHead(
            classification_head=classification_head,
            regression_head=regression_head
        )
        
        self.model = SSD(
            backbone=extractor,
            anchor_generator=anchor_generator,
            size=input_shape[1:3],
            num_classes=num_classes,
            head=head
        )
        
        self.input_shape = input_shape
        self.lr_head = lr_head
        self.lr_extractor = lr_extractor
        self.epochs = epochs
        self.weight_decay = weight_decay
        self.distributed = world_size > 1
        self.train_metric_freq = train_metric_freq

        self.train_map_metric = tm.detection.mean_ap.MeanAveragePrecision(iou_thresholds=[0.5])
        self.val_map_metric = tm.detection.mean_ap.MeanAveragePrecision(iou_thresholds=[0.5])
        self.test_map_metric = tm.detection.mean_ap.MeanAveragePrecision(iou_thresholds=[0.5])
        
    def forward(self, images, targets=None):
        return self.model(images, targets)
    
    def training_step(self, batch, batch_idx):
        images, targets = batch
        outputs = self.model(images)
        loss_dict = self.model(images, targets)
        total_loss = sum(loss for loss in loss_dict.values())

        self.log("train/cls_loss", loss_dict['classification'], prog_bar=True)
        self.log("train/bbox_loss", loss_dict['bbox_regression'], prog_bar=True)
        self.log("train/loss", total_loss, prog_bar=True)
        
        self.train_map_metric.update(outputs, targets)

        return total_loss
        
    def validation_step(self, batch, batch_idx):
        images, targets = batch
        outputs = self.model(images)  # Get predictions
        loss_dict = self.model(images, targets)
        total_loss = sum(loss for loss in loss_dict.values())

        self.log("val/cls_loss", loss_dict['classification'], prog_bar=True)
        self.log("val/bbox_loss", loss_dict['bbox_regression'], prog_bar=True)
        self.log("val/loss", total_loss, prog_bar=True)

        self.val_map_metric.update(outputs, targets)

        return total_loss
    
    def test_step(self, batch, batch_idx):
        images, targets = batch
        outputs = self.model(images)  # Get predictions
        loss_dict = self.model(images, targets)
        total_loss = sum(loss for loss in loss_dict.values())

        # Log test losses
        self.log("test/cls_loss", loss_dict['classification'], prog_bar=True)
        self.log("test/bbox_loss", loss_dict['bbox_regression'], prog_bar=True)
        self.log("test/loss", total_loss, prog_bar=True)

        self.test_map_metric.update(outputs, targets)

        return total_loss
        