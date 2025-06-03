from typing import Tuple, List, Optional
import math
from collections import OrderedDict
from functools import partial

import torch
import torch.nn as nn
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch import Tensor
import pytorch_lightning as pl
import torchmetrics as tm
import torchsummary
import torchvision.transforms.functional as F
from torchvision.models.detection.anchor_utils import DefaultBoxGenerator
from torchvision.models.detection.ssd import SSD
from torchvision.models.detection.ssdlite import SSDLiteHead
import matplotlib.pyplot as plt

from src.constants import IMAGENET_MEAN, IMAGENET_STD

def normal_init(conv_module: nn.Module, mean: float = 0.0, std: float = 0.03, bias: float = 0.0):
    for layer in conv_module.modules():
        if isinstance(layer, nn.Conv2d):
            torch.nn.init.normal_(layer.weight, mean=mean, std=std)
            if layer.bias is not None:
                torch.nn.init.constant_(layer.bias, bias)


class PLBoxGenerator(DefaultBoxGenerator):
    """
    Same functionality as DefaultBoxGenerator, but does not add reciprocal aspect
    ratios.
    """

    def __init__(
        self,
        aspect_ratios: List[List[float]],
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


class SSDLiteMobileNetV3SmallWrapper(nn.Module):

    def __init__(self, backbone: nn.Module, frozen_base: bool = False):
        super().__init__()
        self.features = backbone.features
        if frozen_base:
            print("Training SSDLite with frozen backbone.")
            self.features.requires_grad_(False)

        # For MobileNetV3Small, feature map indices are [1, 3, 6, 9, 12]
        self.feat_idxs = [1, 3, 6, 9, 12]
        self.out_channels = [self.features[i].out_channels for i in self.feat_idxs]

    def forward(self, x):
        """
        Forward pass of the feature extractor. Returns a collection of feature maps,
        each of which can be passed to the SSD head.

        :param x: Input tensor of shape (n, c, h, w)
        :return: Ordered dictionary of feature maps from various blocks

        """
        features = []
        for i, layer in enumerate(self.features):
            x = layer(x)
            if i in self.feat_idxs:
                features.append(x)
        return OrderedDict([(str(i), v) for i, v in enumerate(features)])


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
            min_ratio: float = 0.1,
            max_ratio: float = 0.6,
            aspect_ratios: Optional[List[float]] = None,
            alpha: float = 1.,
            iou_thresh: float = 0.3,
            detections_per_img: int = 50,
            score_thresh: float = 0.01,
            map_iou_thresholds: Optional[List[float]] = None,
            val_log_img_per_batch: int = 1
        ):
        """
        :param extractor: Feature extractor
        :param input_shape: Expected shape of input images (C, H, W)
        :param num_classes: Number of object classes, including background
        :param lr_head: Learning rate for classification head (final layer)
        :param lr_extractor: Learning rate for feature extractor
        :param epochs: Number of epochs
        :param weight_decay: Weight decay for optimizer
        :param frozen_backbone: True if backbone is to be frozen
        :param world_size: Number of devices for distributed training
        :param train_metric_freq: Number of steps after which to log training metrics
        :param min_ratio: Minimum anchor box scale
        :param max_ratio: Maximum anchor box scale
        :param aspect_ratios: Aspect ratios for anchor boxes
        :param alpha: Weight for classification term of loss
        :param iou_thresh: Threshold for intersection between ground truth and anchor boxes
        :param detections_per_img: Number of detections to keep after NMS
        :param score_thresh: Score threshold for postprocessing box detections
        :param map_iou_thresholds: Thresholds over which mAP is computed
        :param val_log_img_per_batch: Number of images to log per validation batch
        """

        super().__init__()
        self.save_hyperparameters()

        self.frozen_backbone = frozen_backbone
        self.iou_thresh = iou_thresh
        self.log_val_img_per_batch = val_log_img_per_batch
        backbone = SSDLiteMobileNetV3SmallWrapper(extractor, frozen_base=self.frozen_backbone).cuda()

        image_size = input_shape[1:3]
        if aspect_ratios is None:
            aspect_ratios = [0.5, 0.75, 1.333, 2.0]
        if map_iou_thresholds is None:
            map_iou_thresholds = [0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]

        n_feat_maps = backbone.out_channels
        anchor_generator = PLBoxGenerator([aspect_ratios for _ in range(len(n_feat_maps))], min_ratio=min_ratio, max_ratio=max_ratio)
        num_anchors = anchor_generator.num_anchors_per_location()
        norm_layer = partial(nn.BatchNorm2d, eps=0.001, momentum=0.03)
        head = SSDLiteHead(n_feat_maps, num_anchors, num_classes, norm_layer)

        self.model = SSD(
            backbone=backbone,
            anchor_generator=anchor_generator,
            size=image_size,
            num_classes=num_classes,
            head=head,
            image_mean=[0., 0., 0.],    # Augmentation pipelines perform normalization
            image_std=[1., 1., 1.],
            iou_thresh=self.iou_thresh,
            detections_per_img=detections_per_img,
            score_thresh=score_thresh
        )
        
        self.input_shape = input_shape
        self.lr_head = lr_head
        self.lr_extractor = lr_extractor
        self.epochs = epochs
        self.weight_decay = weight_decay
        self.distributed = world_size > 1
        self.train_metric_freq = train_metric_freq
        self.alpha = alpha

        self.train_map_metric = tm.detection.mean_ap.MeanAveragePrecision(iou_thresholds=map_iou_thresholds)
        self.val_map_metric = tm.detection.mean_ap.MeanAveragePrecision(iou_thresholds=map_iou_thresholds)
        self.test_map_metric = tm.detection.mean_ap.MeanAveragePrecision(iou_thresholds=map_iou_thresholds)
        self.logged_preds = []
        self.val_log_batches = 2
        
    def forward(self, images, targets=None):
        return self.model(images, targets)
    
    def training_step(self, batch, batch_idx):
        images, targets = batch
        loss_dict = self.model(images, targets)

        total_loss = loss_dict['bbox_regression'] + self.alpha * loss_dict['classification']
        self.log("train/cls_loss", loss_dict['classification'], prog_bar=True, sync_dist=self.distributed)
        self.log("train/bbox_loss", loss_dict['bbox_regression'], prog_bar=True, sync_dist=self.distributed)
        self.log("train/loss", total_loss, prog_bar=True, sync_dist=self.distributed)
        return total_loss
        
    def validation_step(self, batch, batch_idx):
        images, targets = batch
        preds = self.model(images)  # Get predictions

        self.model.train()
        with torch.no_grad():
            loss_dict = self.model(images, targets)
        self.model.eval()
        total_loss = loss_dict['bbox_regression'] + self.alpha * loss_dict['classification']

        self.log("val/cls_loss", loss_dict['classification'], prog_bar=True, on_step=True, on_epoch=True, sync_dist=self.distributed)
        self.log("val/bbox_loss", loss_dict['bbox_regression'], prog_bar=True, on_step=True, on_epoch=True, sync_dist=self.distributed)
        self.log("val/loss", total_loss, prog_bar=True, on_step=True, on_epoch=True, sync_dist=self.distributed)

        self.val_map_metric.update(preds, targets)

        # Log a few box predictions
        self.logged_preds.append(
            (images[:self.log_val_img_per_batch], preds[:self.log_val_img_per_batch], targets[:self.log_val_img_per_batch])
        )
        return total_loss

    def on_validation_epoch_start(self):
        self.logged_preds.clear()

    def on_validation_epoch_end(self):
        map_dict = self.val_map_metric.compute()
        for k in map_dict:
            self.log(f"val/{k}", map_dict[k], prog_bar=True, on_epoch=True, sync_dist=self.distributed)
        self.val_map_metric.reset()

        # Log some validation set box predictions
        for i in range(len(self.logged_preds)):
            image, pred, target = self.logged_preds[i]
            for j in range(len(self.logged_preds[i][0])):
                fig, ax = plt.subplots(1, 1, figsize=(4, 4))
                image = image[j].detach().cpu()
                image = image * torch.tensor(IMAGENET_STD).reshape((3, 1, 1)) + torch.tensor(IMAGENET_MEAN).reshape((3, 1, 1))
                image = F.to_pil_image(image)

                ax.imshow(image)
                # Plot the ground truth bounding boxes
                for box in target[j]['boxes'].cpu():
                    x0, y0, x1, y1 = box.tolist()
                    rect = plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, edgecolor='green', linewidth=2)
                    ax.add_patch(rect)

                # Plot the predicted bounding boxes with confidence above a threshold
                for box, score in zip(pred[j]['boxes'].cpu(), pred[j]['scores'].cpu()):
                    if score > self.iou_thresh:
                        x0, y0, x1, y1 = box.tolist()
                        rect = plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, edgecolor='red', linewidth=2)
                        ax.add_patch(rect)

                ax.axis('off')
                self.logger.experiment.add_figure(f"val/preds_e{self.current_epoch}_b{i}_img{j}", fig, global_step=self.global_step)
                plt.close(fig)
    
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
        for k in map_dict:
            self.log(f"test/{k}", map_dict[k], prog_bar=True, on_epoch=True, sync_dist=self.distributed)
        self.test_map_metric.reset()

    def configure_optimizers(self):
        param_groups = [dict(params=self.model.head.parameters(), lr=self.lr_head)]
        if not self.frozen_backbone:
            param_groups.append(dict(params=self.model.backbone.features.parameters(), lr=self.lr_extractor))
        optimizer = SGD(param_groups, lr=0., momentum=0.9, weight_decay=self.weight_decay)
        scheduler = CosineAnnealingLR(optimizer, T_max=self.epochs, eta_min=0, last_epoch=-1, verbose=True)
        return [optimizer], [{"scheduler": scheduler, "interval": "epoch"}]

    def summary(self):
        torchsummary.summary(self.model.backbone.cuda(), input_size=self.input_shape)
        #torchsummary.summary(self.model.head.cuda(), input_size=self.model.backbone.feature_shapes)
