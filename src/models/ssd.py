from typing import Tuple, Dict, List, Optional
import gc
import math
import time
from collections import OrderedDict
from functools import partial

import torch
import torch.nn as nn
import torchvision.models.detection.ssdlite
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch import Tensor
import pytorch_lightning as pl
import torchmetrics as tm
import torchsummary
from torchvision.models.detection.anchor_utils import DefaultBoxGenerator
from torchvision.models.detection.ssd import SSD
from torchvision.models.detection.ssdlite import SSDLiteHead, _mobilenet_extractor
from torchvision.models.detection import _utils as det_utils


class PLBoxGenerator(DefaultBoxGenerator):
    """
    Same functionality as DefaultBoxGenerator, but does not add reciprocal aspect
    ratios.
    """

    def __init__(
        self,
        aspect_ratios: List[List[int]],
        min_ratio: float = 0.15,
        max_ratio: float = 0.9,
        scales: Optional[List[float]] = None,
        steps: Optional[List[int]] = None,
        clip: bool = True,
    ):
        super().__init__(aspect_ratios, min_ratio, max_ratio, scales, steps, clip)

    def _generate_wh_pairs(
        self, num_outputs: int, dtype: torch.dtype = torch.float32, device: torch.device = torch.device("cpu")
    ) -> List[Tensor]:
        _wh_pairs: List[Tensor] = []
        for k in range(num_outputs):
            # Adding the 2 default width-height pairs for aspect ratio 1 and scale s'k
            s_k = self.scales[k]
            s_prime_k = math.sqrt(self.scales[k] * self.scales[k + 1])
            wh_pairs = [[s_k, s_k], [s_prime_k, s_prime_k]]

            # Adding 2 pairs for each aspect ratio of the feature map k
            for ar in self.aspect_ratios[k]:
                sq_ar = math.sqrt(ar)
                w = self.scales[k] * sq_ar
                h = self.scales[k] / sq_ar
                wh_pairs.extend([[w, h]]) # *** The difference is here ***

            _wh_pairs.append(torch.as_tensor(wh_pairs, dtype=dtype, device=device))
        return _wh_pairs

    def num_anchors_per_location(self) -> List[int]:
        # Estimate num of anchors based on aspect ratios: 2 default boxes + 1 * ratios of feature map.
        return [2 + len(r) for r in self.aspect_ratios]  #  *** Only 1 ratio per feature map


class SSDLiteMobileNetWrapper(nn.Module):

    def __init__(self, backbone, feature_indices):
        super().__init__()
        self.backbone = backbone.features

        # Extract feature maps from different layers
        self.feature_layers = [f'{i}' for i in feature_indices]

        self.feature_shapes = []

    def forward(self, x):
        features = []
        feat_shapes = []
        for name, layer in self.backbone.named_children():
            x = layer(x)
            if name in self.feature_layers:
                features.append(x)
                feat_shapes.append(x.shape[1:])
            if len(self.feature_shapes) == 0:
                self.feature_shapes = feat_shapes
        return OrderedDict([(str(i), v) for i, v in enumerate(features)])  # Return multi-scale features


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
            train_metric_freq: int = 100,
            map_iou_threshold: float = 0.5,
            min_ratio: float = 0.1,
            max_ratio: float = 0.6,
            aspect_ratios: Optional[List[float]] = None,
            alpha: float = 0.25
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

        self.frozen_backbone = frozen_backbone
        norm_layer = partial(nn.BatchNorm2d, eps=0.001, momentum=0.03)

        backbone = SSDLiteMobileNetWrapper(extractor, [1, 3, 6, 9, 12])
        if self.frozen_backbone:
            print("Training SSD with frozen backbone.")
            backbone.requires_grad_(False)
            backbone.eval()

        size = input_shape[1:3]
        if aspect_ratios is None:
            aspect_ratios = [0.5, 0.75, 1.333, 2.0]

        # anchor_generator = DefaultBoxGenerator(
        #     [aspect_ratios for _ in range(6)],
        #     min_ratio=min_ratio,
        #     max_ratio=max_ratio
        # )
        anchor_generator = PLBoxGenerator([aspect_ratios for _ in range(5)], min_ratio=min_ratio, max_ratio=max_ratio)
        out_channels = det_utils.retrieve_out_channels(backbone, size)
        num_anchors = anchor_generator.num_anchors_per_location()
        head = SSDLiteHead(out_channels, num_anchors, num_classes, norm_layer)

        self.model = SSD(
            backbone=backbone,
            anchor_generator=anchor_generator,
            size=input_shape[1:3],
            num_classes=num_classes,
            head=head,
            image_mean=[0., 0., 0.],    # Augmentation pipelines perform normalization
            image_std=[1., 1., 1.],
            iou_thresh=0.6,
            detections_per_img=5,
            score_thresh=0.05
        )
        
        self.input_shape = input_shape
        self.lr_head = lr_head
        self.lr_extractor = lr_extractor
        self.epochs = epochs
        self.weight_decay = weight_decay
        self.distributed = world_size > 1
        self.train_metric_freq = train_metric_freq
        self.alpha = alpha

        self.train_map_metric = tm.detection.mean_ap.MeanAveragePrecision(iou_thresholds=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
        self.val_map_metric = tm.detection.mean_ap.MeanAveragePrecision(iou_thresholds=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
        self.test_map_metric = tm.detection.mean_ap.MeanAveragePrecision(iou_thresholds=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
        
    def forward(self, images, targets=None):
        return self.model(images, targets)
    
    def training_step(self, batch, batch_idx):
        images, targets = batch
        loss_dict = self.model(images, targets)

        total_loss = loss_dict['bbox_regression'] + self.alpha * loss_dict['classification']
        self.log("train/cls_loss", loss_dict['classification'], prog_bar=True, sync_dist=self.distributed)
        self.log("train/bbox_loss", loss_dict['bbox_regression'], prog_bar=True, sync_dist=self.distributed)
        self.log("train/loss", total_loss, prog_bar=True, sync_dist=self.distributed)

        # self.model.eval()
        # with torch.no_grad():
        #     preds = self.model(images)
        # self.model.train()
        # self.train_map_metric.update(preds, targets)
        return total_loss

    def on_train_epoch_end(self):
        # map_dict = self.train_map_metric.compute()
        # #self.log("train/mAP", map_dict['map'], prog_bar=True, on_epoch=True, sync_dist=self.distributed)
        # for k in map_dict:
        #     self.log(f"train/{k}", map_dict[k], prog_bar=True, on_epoch=True, sync_dist=self.distributed)
        #self.log("train/mAP@50", map_dict['map_50'], prog_bar=True, on_epoch=True, sync_dist=self.distributed)
        self.train_map_metric.reset()
        
    def validation_step(self, batch, batch_idx):
        images, targets = batch
        preds = self.model(images)  # Get predictions

        self.model.train()
        with torch.no_grad():
            loss_dict = self.model(images, targets)
        self.model.eval()
        total_loss = loss_dict['bbox_regression'] + self.alpha * loss_dict['classification']

        self.log("val/cls_loss", loss_dict['classification'], prog_bar=True, sync_dist=self.distributed)
        self.log("val/bbox_loss", loss_dict['bbox_regression'], prog_bar=True, sync_dist=self.distributed)
        self.log("val/loss", total_loss, prog_bar=True, sync_dist=self.distributed)

        self.val_map_metric.update(preds, targets)
        return total_loss

    def on_validation_epoch_end(self):
        map_dict = self.val_map_metric.compute()
        print(map_dict)
        #self.log("val/mAP", map_dict['map'], prog_bar=True, on_epoch=True, sync_dist=self.distributed)
        #self.log("val/mAP@50", map_dict['map_50'], prog_bar=True, on_epoch=True, sync_dist=self.distributed)
        for k in map_dict:
            self.log(f"val/{k}", map_dict[k], prog_bar=True, on_epoch=True, sync_dist=self.distributed)
        self.val_map_metric.reset()
    
    def test_step(self, batch, batch_idx):
        images, targets = batch
        preds = self.model(images)  # Get predictions

        self.model.train()
        with torch.no_grad():
            loss_dict = self.model(images, targets)
        self.model.eval()
        total_loss = loss_dict['classification'] + loss_dict['bbox_regression']

        self.log("test/cls_loss", loss_dict['classification'], prog_bar=True, sync_dist=self.distributed)
        self.log("test/bbox_loss", loss_dict['bbox_regression'], prog_bar=True, sync_dist=self.distributed)
        self.log("test/loss", total_loss, prog_bar=True, sync_dist=self.distributed)

        self.test_map_metric.update(preds, targets)
        return total_loss

    def on_test_epoch_end(self):
        map_dict = self.test_map_metric.compute()
        # self.log("test/mAP", map_dict['map'], prog_bar=True, on_epoch=True, sync_dist=self.distributed)
        # self.log("test/mAP@50", map_dict['map_50'], prog_bar=True, on_epoch=True, sync_dist=self.distributed)
        for k in map_dict:
            self.log(f"test/{k}", map_dict[k], prog_bar=True, on_epoch=True, sync_dist=self.distributed)
        self.test_map_metric.reset()

    def configure_optimizers(self):
        param_groups = [dict(params=self.model.head.parameters(), lr=self.lr_head)]
        if not self.frozen_backbone:
            param_groups.append(dict(params=self.model.backbone.parameters(), lr=self.lr_extractor))
        optimizer = SGD(param_groups, lr=0., momentum=0.9, weight_decay=self.weight_decay)
        scheduler = CosineAnnealingLR(optimizer, T_max=self.epochs, eta_min=0, last_epoch=-1, verbose=True)
        return [optimizer], [{"scheduler": scheduler, "interval": "epoch"}]

    def summary(self):
        torchsummary.summary(self.model.backbone.cuda(), input_size=self.input_shape)
        #torchsummary.summary(self.model.head.cuda(), input_size=self.model.backbone.feature_shapes)

        