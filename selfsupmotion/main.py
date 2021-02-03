#!/usr/bin/env python

import argparse
import logging
import os
import sys
import typing
import yaml

from yaml import load
import pytorch_lightning as pl
from pytorch_lightning.loggers import MLFlowLogger
from pytorch_lightning import seed_everything
from pytorch_lightning.utilities import AMPType

from selfsupmotion.train import train
from selfsupmotion.utils.hp_utils import check_and_log_hp
from selfsupmotion.models.model_loader import load_model
from selfsupmotion.models.model_loader import load_optimizer
from selfsupmotion.models.model_loader import load_loss
from selfsupmotion.utils.file_utils import rsync_folder
from selfsupmotion.utils.logging_utils import LoggerWriter, log_exp_details
from selfsupmotion.utils.reproducibility_utils import set_seed

import selfsupmotion.data.objectron.hdf5_parser
from selfsupmotion.models.simsiam import SimSiam

logger = logging.getLogger(__name__)


def main():
    # """Main entry point of the program.
    #
    # Note:
    #     This main.py file is meant to be called using the cli,
    #     see the `examples/local/run.sh` file to see how to use it.
    #
    # """
    # parser = argparse.ArgumentParser()
    # # __TODO__ check you need all the following CLI parameters
    # parser.add_argument('--log', help='log to this file (in addition to stdout/err)')
    # parser.add_argument('--config',
    #                     help='config file with generic hyper-parameters,  such as optimizer, '
    #                          'batch_size, ... -  in yaml format')
    # parser.add_argument('--data', help='path to data', required=True)
    # parser.add_argument('--tmp-folder',
    #                     help='will use this folder as working folder - it will copy the input data '
    #                          'here, generate results here, and then copy them back to the output '
    #                          'folder')
    # parser.add_argument('--output', help='path to outputs - will store files here', required=True)
    # parser.add_argument('--disable-progressbar', action='store_true',
    #                     help='will disable the progressbar while going over the mini-batch')
    # parser.add_argument('--start-from-scratch', action='store_true',
    #                     help='will not load any existing saved model - even if present')
    # parser.add_argument('--debug', action='store_true')
    # args = parser.parse_args()
    #
    # logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    #
    # if not os.path.exists(args.output):
    #     os.makedirs(args.output)
    #
    # if args.tmp_folder is not None:
    #     data_folder_name = os.path.basename(os.path.normpath(args.data))
    #     rsync_folder(args.data, args.tmp_folder)
    #     data_dir = os.path.join(args.tmp_folder, data_folder_name)
    #     output_dir = os.path.join(args.tmp_folder, 'output')
    #     if not os.path.exists(output_dir):
    #         os.makedirs(output_dir)
    # else:
    #     data_dir = args.data
    #     output_dir = args.output
    #
    # # will log to a file if provided (useful for orion on cluster)
    # if args.log is not None:
    #     handler = logging.handlers.WatchedFileHandler(args.log)
    #     formatter = logging.Formatter(logging.BASIC_FORMAT)
    #     handler.setFormatter(formatter)
    #     root = logging.getLogger()
    #     root.setLevel(logging.INFO)
    #     root.addHandler(handler)
    #
    # # to intercept any print statement:
    # sys.stdout = LoggerWriter(logger.info)
    # sys.stderr = LoggerWriter(logger.warning)
    #
    # if args.config is not None:
    #     with open(args.config, 'r') as stream:
    #         hyper_params = load(stream, Loader=yaml.FullLoader)
    # else:
    #     hyper_params = {}
    #
    # # to be done as soon as possible otherwise mlflow will not log with the proper exp. name
    # mlf_logger = MLFlowLogger(
    #     experiment_name=hyper_params.get('exp_name', 'Default')
    # )
    # run(args, data_dir, output_dir, hyper_params, mlf_logger)
    #
    # if args.tmp_folder is not None:
    #     rsync_folder(output_dir + os.path.sep, args.output)

    from pl_bolts.callbacks.ssl_online import SSLOnlineEvaluator

    seed_everything(1234)

    parser = argparse.ArgumentParser()

    # trainer args
    parser = pl.Trainer.add_argparse_args(parser)

    # model args
    parser = SimSiam.add_model_specific_args(parser)
    args = parser.parse_args()

    # pick data
    dm = None

    # init datamodule
    if args.dataset != "objectron":
        raise NotImplementedError

    args.maxpool1 = True
    args.first_conv = True
    args.input_height = 224
    args.gaussian_blur = True
    args.jitter_strength = 1.0
    dm = selfsupmotion.data.objectron.hdf5_parser.ObjectronFramePairDataModule(
        hdf5_path=os.path.join(args.data_dir, "objectron/extract_s5_raw.hdf5"),
        input_height=args.input_height,
        gaussian_blur=args.gaussian_blur,
        jitter_strength=args.jitter_strength,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    args.num_samples = dm.train_sample_count

    model = SimSiam(**args.__dict__)

    # finetune in real-time
    online_evaluator = None
    if args.online_ft:
        # online eval @@@@@@@@@@@ TODO re-check if this can be useful
        online_evaluator = SSLOnlineEvaluator(
            drop_p=0.0,
            hidden_dim=None,
            z_dim=args.hidden_mlp,
            num_classes=dm.num_classes,
            dataset=args.dataset,
        )

    trainer = pl.Trainer.from_argparse_args(
        args,
        sync_batchnorm=True if args.gpus > 1 else False,
        callbacks=[online_evaluator] if args.online_ft else None,
    )

    trainer.fit(model, datamodule=dm)

#
# def run(args, data_dir, output_dir, hyper_params, mlf_logger):
#     """Setup and run the dataloaders, training loops, etc.
#
#     Args:
#         args (list): arguments passed from the cli
#         data_dir (str): path to input folder
#         output_dir (str): path to output folder
#         hyper_params (dict): hyper parameters from the config file
#         mlf_logger (obj): MLFlow logger callback.
#     """
#     # __TODO__ change the hparam that are used from the training algorithm
#     # (and NOT the model - these will be specified in the model itself)
#     logger.info('List of hyper-parameters:')
#     check_and_log_hp(
#         ['architecture', 'batch_size', 'exp_name', 'max_epoch', 'optimizer', 'patience', 'seed'],
#         hyper_params)
#
#     if hyper_params["seed"] is not None:
#         set_seed(hyper_params["seed"])
#
#     log_exp_details(os.path.realpath(__file__), args)
#
#     train_loader, dev_loader = load_data(data_dir, hyper_params)
#     model = load_model(hyper_params)
#     optimizer = load_optimizer(hyper_params, model)
#     loss_fun = load_loss(hyper_params)
#
#     train(model=model, optimizer=optimizer, loss_fun=loss_fun, train_loader=train_loader,
#           dev_loader=dev_loader, patience=hyper_params['patience'], output=output_dir,
#           max_epoch=hyper_params['max_epoch'], use_progress_bar=not args.disable_progressbar,
#           start_from_scratch=args.start_from_scratch, mlf_logger=mlf_logger)


if __name__ == '__main__':
    main()
