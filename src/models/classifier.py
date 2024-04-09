from typing import Tuple

import torch
import torch.nn as nn
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch import Tensor
import pytorch_lightning as pl
import torchmetrics
import torchsummary

class Classifier(pl.LightningModule):

    def __init__(
            self,
            extractor: nn.Module,
            input_shape: Tuple[int, int, int],
            n_classes: int,
            lr: float,
            epochs: int = 10,
            weight_decay: float = 1e-6,
            linear: bool = False
        ):
        """
        Args:
            extractor: Feature extractor
            input_shape: Expected shape of input images (C, H, W)
            n_classes: Number of classes
            lr: Learning rate
            epochs: Number of epochs
            weight_decay: Weight decay for optimizer
        """

        super().__init__()
        self.save_hyperparameters()

        # Define the self-supervised loss
        self.extractor = extractor
        self.linear = linear
        if self.linear:
            self.extractor.requires_grad_(False)
            self.extractor.eval()
        self.input_shape = input_shape
        self.n_classes = n_classes
        self.lr = lr
        self.epochs = epochs
        self.weight_decay = weight_decay

        self.h_dim = self.extractor(torch.randn(*((1,) + input_shape)).cuda()).shape[-1]
        task = 'multiclass' if self.n_classes > 2 else 'binary'
        n_logits = 1 if task == 'binary' else n_classes
        self.head = nn.Linear(self.h_dim, n_logits)
        self.output_activation = nn.Sigmoid() if task == 'binary' else nn.Softmax()

        self.loss = nn.BCELoss() if n_classes == 2 else nn.NLLLoss()

        self.accuracy = torchmetrics.Accuracy(task, num_classes=n_classes)
        self.precision = torchmetrics.Precision(task, num_classes=n_classes)
        self.recall = torchmetrics.Recall(task, num_classes=n_classes)
        self.auroc = torchmetrics.AUROC(task, num_classes=n_classes)
        self.metrics = {
            'accuracy': self.accuracy,
            'precision': self.precision,
            'recall': self.recall,
            'AUROC': self.auroc
        }

    def forward(self, x: Tensor) -> Tensor:

        # Compute feature representations
        h = self.extractor(x)

        # Compute logits
        logits = self.head(h)
        probs = self.output_activation(logits)
        return probs

    def training_step(self, batch, batch_idx):

        # Forward pass
        x, y = batch
        y_hat = self.forward(x)
        loss = self.loss(y_hat, y)

        # Log the loss and metrics
        self.log('train/loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        for m in self.metrics:
            self.metrics[m](y_hat, y)
            self.log(f'train/{m}', self.metrics[m], on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):

        # Forward pass
        x, y = batch
        y_hat = self.forward(x)
        loss = self.loss(y_hat, y)

        # Log the loss and metrics
        self.log('val/loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        for m in self.metrics:
            self.metrics[m](y_hat, y)
            self.log(f'val/{m}', self.metrics[m], on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = SGD(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = CosineAnnealingLR(optimizer, T_max=self.epochs, eta_min=0, last_epoch=-1, verbose=True)
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]

    def summary(self):
        torchsummary.summary(self.extractor, input_size=self.input_shape)
        torchsummary.summary(self.projector, input_size=(self.h_dim,))
