
import torch
from torch import nn
from torch.nn import functional as F
import pytorch_lightning as pl
from unet import UNet
from torch.optim import AdamW
from metrics import mIOU

class UNetWrapper(pl.LightningModule):
    def __init__(self, config_dict = dict()):
        super().__init__()
        self.unet = UNet(3, 2, bilinear=True)
        self.lr = config_dict.get('lr', 1e-3)
        self.num_classes = config_dict.get('num_classes', 2)
        self.ignore_index = config_dict.get('ignore_index', -1)

        self.train_miou_metric = mIOU(self.num_classes, self.ignore_index, nan_to_num=0.0)
        self.val_miou_metric = self.train_miou_metric.clone()

    def forward(self, x):
        return self.unet(x)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        img = train_batch['image']
        label = train_batch['label']
        pred = self.unet(img)
        loss = F.cross_entropy(pred, label, reduction="mean")
        self.log('train_loss', loss)
        if batch_idx % 4 == 0:
            miou = self.train_miou_metric(pred, label)        
            self.log('train_miou_step', miou, prog_bar = True)
        return loss

    def training_epoch_end(self, outputs):
        # log epoch metric
        self.log('train_miou_epoch', self.train_miou_metric.compute())
        self.train_miou_metric.reset()
        super().training_epoch_end(outputs)

    def validation_epoch_end(self, outputs):
        # log epoch metric
        self.log('val_miou_epoch', self.val_miou_metric.compute())
        self.val_miou_metric.reset()
        super().validation_epoch_end(outputs)
        
    def validation_step(self, val_batch, batch_idx):
        img = val_batch['image']
        label = val_batch['label']
        pred = self.unet(img)
        loss = F.cross_entropy(pred, label, reduction="mean")
        self.log('val_loss', loss)
        miou = self.val_miou_metric(pred, label)        
        self.log('val_miou_step', miou)
