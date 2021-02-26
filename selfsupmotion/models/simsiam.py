import logging
import math
import typing

import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.utilities import AMPType
from torch.nn.modules.linear import Identity
from torch.optim.optimizer import Optimizer
import torchvision

from pl_bolts.models.self_supervised.resnets import resnet18, resnet50
from torchvision.models.shufflenetv2 import shufflenet_v2_x1_0
#from pl_bolts.models.self_supervised.simsiam.models import SiameseArm
from pl_bolts.optimizers.lars_scheduling import LARSWrapper

import torch.nn.functional as F 
import torch.nn as nn
from typing import Optional, Tuple

from selfsupmotion.utils.logging_utils import save_mosaic
import selfsupmotion.zero_shot_pose as zsp

logger = logging.getLogger(__name__)

#Credit: https://github.com/PatrickHua/SimSiam
class ProjectionMLP(nn.Module):
    def __init__(self, in_dim, hidden_dim=2048, out_dim=2048):
        super().__init__()
        ''' page 3 baseline setting
        Projection MLP. The projection MLP (in f) has BN ap-
        plied to each fully-connected (fc) layer, including its out- 
        put fc. Its output fc has no ReLU. The hidden fc is 2048-d. 
        This MLP has 3 layers.
        '''
        self.layer1 = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        self.layer2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        self.layer3 = nn.Sequential(
            nn.Linear(hidden_dim, out_dim),
            nn.BatchNorm1d(hidden_dim)
        )
        self.num_layers = 3
    def set_layers(self, num_layers):
        self.num_layers = num_layers

    def forward(self, x):
        if self.num_layers == 3:
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
        elif self.num_layers == 2:
            x = self.layer1(x)
            x = self.layer3(x)
        else:
            raise Exception
        return x 

#Credit: https://github.com/PatrickHua/SimSiam
class PredictionMLP(nn.Module):
    def __init__(self, in_dim=2048, hidden_dim=512, out_dim=2048): # bottleneck structure
        super().__init__()
        ''' page 3 baseline setting
        Prediction MLP. The prediction MLP (h) has BN applied 
        to its hidden fc layers. Its output fc does not have BN
        (ablation in Sec. 4.4) or ReLU. This MLP has 2 layers. 
        The dimension of h’s input and output (z and p) is d = 2048, 
        and h’s hidden layer’s dimension is 512, making h a 
        bottleneck structure (ablation in supplement). 
        '''
        self.layer1 = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        self.layer2 = nn.Linear(hidden_dim, out_dim)
        """
        Adding BN to the output of the prediction MLP h does not work
        well (Table 3d). We find that this is not about collapsing. 
        The training is unstable and the loss oscillates.
        """

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return x 

class MLP(nn.Module):
    def __init__(self, input_dim: int = 2048, hidden_size: int = 4096, output_dim: int = 256) -> None:
        super().__init__()
        self.output_dim = output_dim
        self.input_dim = input_dim
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_size, bias=False),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, output_dim, bias=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.model(x)
        return x

class SiameseArm(nn.Module):

    def __init__(
        self,
        encoder: Optional[nn.Module] = None,
        input_dim: int = 2048,
        hidden_size: int = 4096,
        output_dim: int = 256,
        hua_mlp = True,
    ) -> None:
        super().__init__()

        if encoder is None:
            raise ValueError("Please provide an encoder.")
        # Encoder
        self.encoder = encoder
        #input_dim = self.encoder.fc.in_features
        # Projector
        if hua_mlp: #Using Patrick Hua interpretation of SimSiam.
            #Will use an additional hidden layer. Linear layer will have bias.
            self.projector = ProjectionMLP(input_dim, output_dim, output_dim)
            self.predictor = PredictionMLP(output_dim, hidden_size, output_dim)
        else:       #Pytorch Lighting interepreation of SimSiam
            self.projector = MLP(input_dim, hidden_size, output_dim)
            self.predictor = MLP(output_dim, hidden_size, output_dim)
        # Predictor
        

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if type(self.encoder)==torchvision.models.shufflenetv2.ShuffleNetV2:
            y = self.encoder(x)
        else:
            y = self.encoder(x)[0]
        z = self.projector(y)
        h = self.predictor(z)
        return y, z, h

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
        self.coords_channels = hyper_params.get("coords_channels",0)

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

        self.accumulate_grad_batches_custom = hyper_params.get("accumulate_grad_batches_custom",1)

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

        #self.log("val_loss",0)
        #self.log("train_loss", 0)



    def init_model(self):
        assert self.backbone in ["resnet18", "resnet50", "shufflenet_v2_x1_0"]
        if self.backbone == "resnet18":
            backbone = resnet18
            backbone_network = backbone(first_conv=self.first_conv, maxpool1=self.maxpool1, return_all_feature_maps=False)
            self.feature_dim = backbone_network.fc.in_features

        elif self.backbone == "resnet50":
            backbone = resnet50
            backbone_network = backbone(first_conv=self.first_conv, maxpool1=self.maxpool1, return_all_feature_maps=False)
            self.feature_dim = backbone_network.fc.in_features
            if self.coords_channels>0:
                backbone_network.conv1=nn.Conv2d(3+self.coords_channels,64,kernel_size=(7,7),stride=(2,2),padding=(3,3),bias=False)
        elif self.backbone == "shufflenet_v2_x1_0":
            backbone = shufflenet_v2_x1_0
            backbone_network = backbone()
            self.feature_dim = backbone_network.fc.in_features
            backbone_network.fc = Identity()
        else:
            raise ValueError(f"Unsupported backbone: {self.backbone}")
        
        self.online_network = SiameseArm(
            backbone_network, input_dim=self.feature_dim, hidden_size=self.hidden_mlp, output_dim=self.feat_dim
        )
        #max_batch = math.ceil(self.num_samples/self.batch_size)
        encoder, projector = self.online_network.encoder, self.online_network.projector
        
        self.train_features = torch.zeros((self.num_samples, self.feature_dim))
        self.train_meta = []
        self.train_targets = -torch.ones((self.num_samples))
        self.valid_features = torch.zeros((self.num_samples_valid, self.feature_dim))
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
        assert len(batch["OBJ_CROPS"]) == 2
        img_1, img_2 = batch["OBJ_CROPS"]

        if batch_idx==0:
            save_mosaic("img_1_train.jpg", img_1)
            save_mosaic("img_2_train.jpg", img_2)
            
        #assert img_1.shape==torch.Size([32, 3, 224, 224])
        uid = batch["UID"]
        y = batch["CAT_ID"]

        if self.cuda_train_features is not None:
            self.cuda_train_features = None #Free GPU memory
        # Image 1 to image 2 loss
        f1, z1, h1 = self.online_network(img_1.float())
        f2, z2, h2 = self.online_network(img_2.float())
        loss = self.cosine_similarity(h1, z2) / 2 + self.cosine_similarity(h2, z1) / 2

        base = batch_idx*self.batch_size
        train_features= F.normalize(f1.detach(), dim=1).cpu()
        #assert train_features.shape == torch.Size([32, 2048])
        self.train_meta+=uid
        self.train_features[base:base+train_features.shape[0]]=train_features
        self.train_targets[base:base+train_features.shape[0]]=y
        # log results
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        assert len(batch["OBJ_CROPS"]) == 2
        img_1, img_2 = batch["OBJ_CROPS"]

        if batch_idx==0:
            save_mosaic("img_1_val.jpg", img_1)
            save_mosaic("img_2_val.jpg", img_2)
            if self.cuda_train_features is None: #Transfer to GPU once.
                self.cuda_train_features = self.train_features.half().cuda()

        uid = batch["UID"]
        y = batch["CAT_ID"]

        # Image 1 to image 2 loss
        f1, z1, h1 = self.online_network(img_1.float())
        f2, z2, h2 = self.online_network(img_2.float())

        loss = self.cosine_similarity(h1, z2) / 2 + self.cosine_similarity(h2, z1) / 2

        self.valid_meta+=uid
        base = batch_idx*self.batch_size

        valid_features = F.normalize(f1, dim=1).detach()

        similarity = torch.mm(valid_features, self.cuda_train_features.T)
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
        #if self.trainer.amp_backend == AMPType.NATIVE:
        #    optimizer_closure()
        #    self.trainer.scaler.step(optimizer)
        if ((batch_idx+1)%self.accumulate_grad_batches_custom)==0:
            if self.trainer.amp_backend == AMPType.APEX:
                optimizer_closure()
                optimizer.step()
            else:
                optimizer.step(closure=optimizer_closure)
