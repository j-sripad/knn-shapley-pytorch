import copy 

import torch 
import torch.nn as nn 
import torch.nn.functional as F 
from torchmetrics.functional import iou, accuracy
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor

from models.unet import UNet 


class DShap: 

    def  __init__(self, data_module, model, metric="accuracy", measure=None, device="cuda"):
        self.data_module = data_module
        self.num_classes = data_module.num_classes 
        self.metric = metric
        self.model = model
        self.measure = measure
        self.device = device
        self.random_score = self.init_score(self.metric)

        checkpoint_callback = ModelCheckpoint(
            monitor="val_loss_epoch",
            dirpath="./checkpoints/",
            filename="net-{epoch:02d}-{val_loss_epoch:.2f}",
            save_top_k=-1,
        )

        checkpoint_callback.FILE_EXTENSION = ".pth.tar"
        lr_monitor = LearningRateMonitor(logging_interval="step")

        self.callbacks = [checkpoint_callback, lr_monitor]

    def init_score(self, metric):
        """ Gives the value of an initial untrained model."""
        
        if metric == "iou":
            score = 0
            n = 0
            for X, y in self.data_module.test_dataloader():
                score += iou(torch.randint(0, self.num_classes, y.shape), y)
                n += 1
            return score / n
        elif metric == "accuracy": 
            score = 0 
            n = 0
            for X, y in self.data_module.test_dataloader():
                score += accuracy(torch.randint(0, self.num_classes, y.shape), y)
                n += 1
            return score / n 

    def run(self, max_epochs=10): 
        self.restart_model() 
        if self.device == "cuda":
            self.trainer = pl.Trainer(gpus=1, max_epochs=max_epochs, callbacks=self.callbacks) 
            self.trainer.fit(self.model, self.data_module)
        else:
            self.trainer = pl.Trainer(max_epochs=max_epochs, callbacks=self.callbacks) 
            self.trainer.fit(self.model, self.data_module)

        if self.measure is not None: 
            return self.measure.score(self.data_module, self.model)

    def restart_model(self): 
        self.model = copy.deepcopy(self.model)