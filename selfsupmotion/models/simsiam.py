import logging
import math
import typing

import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.utilities import AMPType
from torch.nn import functional as F
from torch.optim.optimizer import Optimizer

from pl_bolts.models.self_supervised.resnets import resnet18, resnet50
from pl_bolts.models.self_supervised.simsiam.models import SiameseArm
from pl_bolts.optimizers.lars_scheduling import LARSWrapper

import torch.nn.functional as F 

logger = logging.getLogger(__name__)


class SimSiam(pl.LightningModule):

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
        self.num_samples_valid = hyper_params.get("num_samples_valid")
        self.batch_size = hyper_params.get("batch_size")

        self.hidden_mlp = hyper_params.get("hidden_mlp", 2048)
        self.feat_dim = hyper_params.get("feat_dim", 128)
        self.first_conv = hyper_params.get("first_conv", True)
        self.maxpool1 = hyper_params.get("maxpool1", True)

        self.optim = hyper_params.get("optimizer", "adam")
        self.lars_wrapper = hyper_params.get("lars_wrapper", True)
        self.exclude_bn_bias = hyper_params.get("exclude_bn_bias", False)
        self.weight_decay = hyper_params.get("weight_decay", 1e-6)
        self.temperature = hyper_params.get("temperature", 0.1)

        self.start_lr = hyper_params.get("start_lr", 0.)
        self.final_lr = hyper_params.get("final_lr", 1e-6)
        self.learning_rate = hyper_params.get("learning_rate", 1e-3)
        self.warmup_epochs = hyper_params.get("warmup_epochs", 10)
        self.max_epochs = hyper_params.get("max_epochs", 100)

        self.init_model()

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

        backbone_network = backbone(first_conv=self.first_conv, maxpool1=self.maxpool1, return_all_feature_maps=False)
        self.online_network = SiameseArm(
            backbone_network, input_dim=self.hidden_mlp, hidden_size=self.hidden_mlp, output_dim=self.feat_dim
        )
        #max_batch = math.ceil(self.num_samples/self.batch_size)
        encoder, projector = self.online_network.encoder, self.online_network.projector
        self.train_features = torch.zeros((self.num_samples,projector.input_dim))
        self.train_meta = []
        self.train_targets = -torch.ones((self.num_samples))
        self.valid_features = torch.zeros((self.num_samples_valid, projector.input_dim))
        self.valid_meta = []
        self.cuda_train_features = None
        

    def forward(self, x):
        y, _, _ = self.online_network(x)
        return y

    def cosine_similarity(self, a, b, version="simplified"):
        if version == "original":
            b = b.detach()  # stop gradient of backbone + projection mlp
            a = F.normalize(a, dim=-1)
            b = F.normalize(b, dim=-1)
            sim = -1 * (a * b).sum(-1).mean()
        elif version=="simplified":
            sim = -F.cosine_similarity(a, b.detach(), dim=-1).mean()
        else:
            raise ValueError(f"Unsupported cosine similarity version: {version}")
        return sim

    def training_step(self, batch, batch_idx):
        (img_1, img_2, meta), y = batch

        if self.cuda_train_features is not None:
            self.cuda_train_features = None #Free GPU memory
        # Image 1 to image 2 loss
        f1, z1, h1 = self.online_network(img_1)
        f2, z2, h2 = self.online_network(img_2)
        loss = self.cosine_similarity(h1, z2) / 2 + self.cosine_similarity(h2, z1) / 2

        base = batch_idx*self.batch_size
        train_features= F.normalize(f1.detach(), dim=1).cpu()
        self.train_meta+=meta
        self.train_features[base:base+len(img_1)]=train_features
        self.train_targets[base:base+len(img_1)]=y
        # log results
        self.log("train_loss", loss)

        return loss

    def validation_step(self, batch, batch_idx):
        (img_1, img_2, meta), y = batch

        # Image 1 to image 2 loss
        f1, z1, h1 = self.online_network(img_1)
        f2, z2, h2 = self.online_network(img_2)
        if self.cuda_train_features is None: #Transfer to GPU once.
            self.cuda_train_features = self.train_features.cuda()

        loss = self.cosine_similarity(h1, z2) / 2 + self.cosine_similarity(h2, z1) / 2

        self.valid_meta+=meta
        base = batch_idx*self.batch_size

        valid_features = F.normalize(f1, dim=1).detach()

        similarity = torch.mm(valid_features, self.train_features.cuda().T)
        targets_idx= torch.argmax(similarity,axis=1).cpu()
        neighbor_targets = self.train_targets[targets_idx]
        match_count = (neighbor_targets==y.cpu()).sum()
        accuracy = match_count/len(neighbor_targets)

        self.valid_features[base:base+len(img_1)]=valid_features

        # log results
        self.log("val_loss", loss)
        self.log("val_accuracy", accuracy)

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

        predictor_prefix = ('encoder')
        backbone_and_encoder_parameters = [param for name, param in self.online_network.encoder.named_parameters()]
        backbone_and_encoder_parameters+= [param for name, param in self.online_network.projector.named_parameters()]
        lr = self.learning_rate
        params = [{
            'name': 'base',
            'params': backbone_and_encoder_parameters,
            'lr': lr
            },{
                'name': 'predictor',
                'params': [param for name, param in self.online_network.predictor.named_parameters()],
                'lr': lr
            }]
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
                if param_group["name"]=="predictor":
                    param_group["lr"] = self.learning_rate
                else:
                    param_group["lr"] = self.lr_schedule[self.trainer.global_step]
            #param_group[0]["lr"]

        # log LR (LearningRateLogger callback doesn't work with LARSWrapper)
        self.log('learning_rate', self.lr_schedule[self.trainer.global_step], on_step=True, on_epoch=False)

        # from lightning
        if self.trainer.amp_backend == AMPType.NATIVE:
            optimizer_closure()
            self.trainer.scaler.step(optimizer)
        elif self.trainer.amp_backend == AMPType.APEX:
            optimizer_closure()
            optimizer.step()
        else:
            optimizer.step(closure=optimizer_closure)
