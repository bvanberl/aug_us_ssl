from typing import Tuple, Dict

import torch
import torch.nn as nn
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch import Tensor
import pytorch_lightning as pl
import torchmetrics as tm
import torchsummary

def normal_init_linear(
        in_dim: int,
        out_dim: int,
        w_mean: float = 0.,
        w_std: float = 0.01
):
    """
    Initializes a fully connected layer with weights randomly
    drawn from a Gaussian distribution with specified
    weight mean and std dev. The layer is initialized with zero
    bias as well.
    :param in_dim: Dimension of input
    :param out_dim: Dimension of output
    :param w_mean: Mean for weight matrix
    :param w_std: Std for weight matrix
    :return: Linear layer
    """
    layer = nn.Linear(in_dim, out_dim)
    layer.weight.data.normal_(mean=w_mean, std=w_std)
    layer.bias.data.zero_()
    return layer

class Classifier(pl.LightningModule):

    def __init__(
            self,
            extractor: nn.Module,
            input_shape: Tuple[int, int, int],
            n_classes: int,
            lr_head: float,
            lr_extractor: float,
            epochs: int = 10,
            weight_decay: float = 1e-6,
            linear: bool = False
        ):
        """
        Args:
            extractor: Feature extractor
            input_shape: Expected shape of input images (C, H, W)
            n_classes: Number of classes
            lr_head: Learning rate for classification head (final layer)
            lr_extractor: Learning rate for feature extractor
            epochs: Number of epochs
            weight_decay: Weight decay for optimizer
        """

        super().__init__()
        self.save_hyperparameters()

        self.extractor = extractor
        self.linear = linear
        if self.linear:
            self.extractor.requires_grad_(False)
            self.extractor.eval()
        self.input_shape = input_shape
        self.n_classes = n_classes
        self.lr_head = lr_head
        self.lr_extractor = lr_extractor
        self.epochs = epochs
        self.weight_decay = weight_decay

        self.h_dim = self.extractor(torch.randn(*((1,) + input_shape)).cuda()).shape[-1]
        task = 'multiclass' if self.n_classes > 2 else 'binary'
        n_logits = 1 if task == 'binary' else n_classes

        linear = normal_init_linear(self.h_dim, n_logits)
        output_activation = nn.Sigmoid() if task == 'binary' else nn.Softmax()
        self.head = nn.Sequential(linear, output_activation)

        self.loss = nn.BCELoss() if n_classes == 2 else nn.NLLLoss()

        metrics = tm.MetricCollection([
            tm.Accuracy(task, num_classes=n_classes),
            tm.Precision(task, num_classes=n_classes),
            tm.Recall(task, num_classes=n_classes),
            tm.F1Score(task, num_classes=n_classes),
            tm.AUROC(task, num_classes=n_classes),
        ])
        self.train_metrics = metrics.clone(prefix='train/')
        self.val_metrics = metrics.clone(prefix='val/')

    def forward(self, x: Tensor) -> Tensor:

        # Compute feature representations
        h = self.extractor(x)

        # Compute output confidences
        y_hat = self.head(h)
        return y_hat

    def training_step(self, batch, batch_idx):

        # Forward pass
        x, y = batch
        y_hat = self.forward(x)
        loss = self.loss(y_hat, y)

        # Log the loss and metrics
        self.log('train/loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        metrics_dict = self.train_metrics(y_hat, y)
        self.log_dict(metrics_dict, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):

        # Forward pass
        x, y = batch
        y_hat = self.forward(x)
        loss = self.loss(y_hat, y)

        # Log the loss and metrics
        self.log('val/loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        metrics_dict = self.val_metrics(y_hat, y)
        self.log_dict(metrics_dict, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        param_groups = [dict(params=self.head.parameters(), lr=self.lr_head)]
        if not self.linear:
            param_groups.append(dict(params=self.extractor.parameters(), lr=self.lr_extractor))
        optimizer = SGD(param_groups, lr=0., momentum=0.9, weight_decay=self.weight_decay)
        scheduler = CosineAnnealingLR(optimizer, T_max=self.epochs, eta_min=0, last_epoch=-1, verbose=True)
        return [optimizer], [{"scheduler": scheduler, "interval": "epoch"}]

    def summary(self):
        torchsummary.summary(self.extractor, input_size=self.input_shape)
        torchsummary.summary(self.head, input_size=(self.h_dim,))
