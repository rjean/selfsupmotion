import logging
import math
import typing

import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.utilities import AMPType
from torch.optim.optimizer import Optimizer

from pl_bolts.models.self_supervised.resnets import resnet18, resnet50
from pl_bolts.optimizers.lars_scheduling import LARSWrapper

logger = logging.getLogger(__name__)
from selfsupmotion.utils.logging_utils import save_mosaic


class KeypointsRegressor(pl.LightningModule):

    def __init__(
            self,
            hyper_params: typing.Dict[typing.AnyStr, typing.Any],
    ):
        super().__init__()
        self.save_hyperparameters(hyper_params)

        self.gpus = hyper_params.get("gpus")
        self.num_nodes = hyper_params.get("num_nodes", 1)
        self.backbone = hyper_params.get("backbone", "resnet50")
        self.num_samples = hyper_params.get("num_samples")
        self.batch_size = hyper_params.get("batch_size")

        #self.hidden_mlp = hyper_params.get("hidden_mlp", 2048)
        #self.feat_dim = hyper_params.get("feat_dim", 128)
        self.first_conv = hyper_params.get("first_conv", True)
        self.maxpool1 = hyper_params.get("maxpool1", True)
        self.dropout = hyper_params.get("dropout", 0.2)
        self.input_height = hyper_params.get("input_height", 224)

        self.optim = hyper_params.get("optimizer", "adam")
        self.lars_wrapper = hyper_params.get("lars_wrapper", True)
        self.exclude_bn_bias = hyper_params.get("exclude_bn_bias", False)
        self.weight_decay = hyper_params.get("weight_decay", 1e-6)
        #self.temperature = hyper_params.get("temperature", 0.1)

        self.start_lr = hyper_params.get("start_lr", 0.)
        self.final_lr = hyper_params.get("final_lr", 1e-6)
        self.learning_rate = hyper_params.get("learning_rate", 1e-3)
        self.warmup_epochs = hyper_params.get("warmup_epochs", 10)
        self.max_epochs = hyper_params.get("max_epochs", 250)

        self.init_model()
        self.loss_fn = torch.nn.MSELoss()

        # compute iters per epoch
        nb_gpus = len(self.gpus) if isinstance(self.gpus, (list, tuple)) else self.gpus
        assert isinstance(nb_gpus, int)
        global_batch_size = self.num_nodes * nb_gpus * self.batch_size if nb_gpus > 0 else self.batch_size
        self.train_iters_per_epoch = self.num_samples // global_batch_size

        # define LR schedule
        warmup_lr_schedule = np.linspace(
            self.start_lr, self.learning_rate, self.train_iters_per_epoch * self.warmup_epochs
        )
        iters = np.arange(self.train_iters_per_epoch * (self.max_epochs - self.warmup_epochs))
        cosine_lr_schedule = np.array([
            self.final_lr + 0.5 * (self.learning_rate - self.final_lr) *
            (1 + math.cos(math.pi * t / (self.train_iters_per_epoch * (self.max_epochs - self.warmup_epochs))))
            for t in iters
        ])

        self.lr_schedule = np.concatenate((warmup_lr_schedule, cosine_lr_schedule))

    def init_model(self):
        assert self.backbone in ["resnet18", "resnet50"]
        if self.backbone == "resnet18":
            backbone = resnet18
        else:
            backbone = resnet50

        self.encoder = backbone(
            first_conv=self.first_conv,
            maxpool1=self.maxpool1,
            return_all_feature_maps=False,
        )
        if self.dropout is not None:  # @@@@ experiment with this
            self.decoder = torch.nn.Sequential(
                torch.nn.Dropout(p=self.dropout),
                torch.nn.Linear(2048, 18, bias=True),
            )
        else:
            self.decoder = torch.nn.Sequential(
                torch.nn.Linear(2048, 1024, bias=False),
                torch.nn.BatchNorm1d(1024),
                torch.nn.ReLU(inplace=True),
                torch.nn.Linear(1024, 18, bias=True),
            )

    def forward(self, x):
        return self.decoder(x)

    def training_step(self, batch, batch_idx):
        if batch_idx==0:
            save_mosaic("train_kpts.png",batch["OBJ_CROPS"][0])

        loss = None
        for img, pts in zip(batch["OBJ_CROPS"], batch["POINTS"]):

            embd = self.encoder(img)[0]
            preds = self.decoder(embd).view(pts.shape)
            curr_loss = self.loss_fn(preds, pts / self.input_height)
            if loss is None:
                loss = curr_loss
            else:
                loss += curr_loss
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        if batch_idx==0:
            save_mosaic("val_kpts.png",batch["OBJ_CROPS"][0])
        loss = None
        for img, pts in zip(batch["OBJ_CROPS"], batch["POINTS"]):
            embd = self.encoder(img)[0]
            preds = self.decoder(embd).view(pts.shape)
            curr_loss = self.loss_fn(preds, pts / self.input_height)
            if loss is None:
                loss = curr_loss
            else:
                loss += curr_loss
        self.log("val_loss", loss)
        return loss

    def exclude_from_wt_decay(self, named_params, weight_decay, skip_list=['bias', 'bn']):
        params = []
        excluded_params = []

        for name, param in named_params:
            if not param.requires_grad:
                continue
            elif any(layer_name in name for layer_name in skip_list):
                excluded_params.append(param)
            else:
                params.append(param)

        return [
            {
                'params': params,
                'weight_decay': weight_decay
            },
            {
                'params': excluded_params,
                'weight_decay': 0.
            },
        ]

    def configure_optimizers(self):
        if self.exclude_bn_bias:
            params = self.exclude_from_wt_decay(self.named_parameters(), weight_decay=self.weight_decay)
        else:
            params = self.parameters()

        if self.optim == 'sgd':
            optimizer = torch.optim.SGD(params, lr=self.learning_rate, momentum=0.9, weight_decay=self.weight_decay)
        elif self.optim == 'adam':
            optimizer = torch.optim.Adam(params, lr=self.learning_rate, weight_decay=self.weight_decay)

        if self.lars_wrapper:
            optimizer = LARSWrapper(
                optimizer,
                eta=0.001,  # trust coefficient
                clip=False
            )

        return optimizer

    def optimizer_step(
            self,
            epoch: int,
            batch_idx: int,
            optimizer: Optimizer,
            optimizer_idx: int,
            optimizer_closure: typing.Optional[typing.Callable] = None,
            on_tpu: bool = False,
            using_native_amp: bool = False,
            using_lbfgs: bool = False,
    ) -> None:
        # warm-up + decay schedule placed here since LARSWrapper is not optimizer class
        # adjust LR of optim contained within LARSWrapper
        if self.lars_wrapper:
            for param_group in optimizer.optim.param_groups:
                param_group["lr"] = self.lr_schedule[self.trainer.global_step]
        else:
            for param_group in optimizer.param_groups:
                param_group["lr"] = self.lr_schedule[self.trainer.global_step]

        # log LR (LearningRateLogger callback doesn't work with LARSWrapper)
        self.log('learning_rate', self.lr_schedule[self.trainer.global_step], on_step=True, on_epoch=False)

        # from lightning
        if self.trainer.amp_backend == AMPType.APEX:
            optimizer_closure()
            optimizer.step()
        else:
            optimizer.step(closure=optimizer_closure)
