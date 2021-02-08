#!/usr/bin/env python

import argparse
import logging
import os
import shutil
import sys
import yaml

from yaml import load
import pytorch_lightning as pl
from pytorch_lightning.loggers import MLFlowLogger

from selfsupmotion.train import train
from selfsupmotion.utils.hp_utils import check_and_log_hp
from selfsupmotion.models.model_loader import load_model
from selfsupmotion.utils.file_utils import rsync_folder
from selfsupmotion.utils.logging_utils import LoggerWriter, log_exp_details
from selfsupmotion.utils.reproducibility_utils import set_seed

import selfsupmotion.data.objectron.hdf5_parser
import selfsupmotion.data.objectron.file_datamodule

logger = logging.getLogger(__name__)


def main():
    """Main entry point of the program.

    Note:
        This main.py file is meant to be called using the cli,
        see the `examples/local/run.sh` file to see how to use it.

    """
    parser = argparse.ArgumentParser()
    # __TODO__ check you need all the following CLI parameters
    parser.add_argument('--log', help='log to this file (in addition to stdout/err)')
    parser.add_argument('--config',
                        help='config file with generic hyper-parameters,  such as optimizer, '
                             'batch_size, ... -  in yaml format')
    parser.add_argument('--data', help='path to data', required=True)
    parser.add_argument('--data-module', default="hdf5", help="Data module to use. file or hdf5")
    parser.add_argument('--tmp-folder',
                        help='will use this folder as working folder - it will copy the input data '
                             'here, generate results here, and then copy them back to the output '
                             'folder')
    parser.add_argument('--output', help='path to outputs - will store files here', required=True)
    parser.add_argument('--disable-progressbar', action='store_true',
                        help='will disable the progressbar while going over the mini-batch')
    parser.add_argument('--start-from-scratch', action='store_true',
                        help='will not load any existing saved model - even if present')
    parser.add_argument('--debug', action='store_true')

    parser = pl.Trainer.add_argparse_args(parser)

    args = parser.parse_args()

    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    if not os.path.exists(args.output):
        os.makedirs(args.output)

    if args.tmp_folder is not None:
        data_folder_name = os.path.basename(os.path.normpath(args.data))
        rsync_folder(args.data, args.tmp_folder)
        data_dir = os.path.join(args.tmp_folder, data_folder_name)
        output_dir = os.path.join(args.tmp_folder, 'output')
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
    else:
        data_dir = args.data
        output_dir = args.output

    # will log to a file if provided (useful for orion on cluster)
    if args.log is not None:
        handler = logging.handlers.WatchedFileHandler(args.log)
        formatter = logging.Formatter(logging.BASIC_FORMAT)
        handler.setFormatter(formatter)
        root = logging.getLogger()
        root.setLevel(logging.INFO)
        root.addHandler(handler)

    # to intercept any print statement:
    sys.stdout = LoggerWriter(logger.info)
    sys.stderr = LoggerWriter(logger.warning)

    assert args.config is not None
    with open(args.config, 'r') as stream:
        hyper_params = load(stream, Loader=yaml.FullLoader)
    exp_name = hyper_params["exp_name"]
    output_dir = os.path.join(output_dir, exp_name)
    os.makedirs(output_dir, exist_ok=True)
    shutil.copyfile(args.config, os.path.join(output_dir, "config.backup"))
    mlf_logger = MLFlowLogger(experiment_name=exp_name)
    run(args, data_dir, output_dir, hyper_params, mlf_logger)
    if args.tmp_folder is not None:
        rsync_folder(output_dir + os.path.sep, args.output)


def run(args, data_dir, output_dir, hyper_params, mlf_logger):
    """Setup and run the dataloaders, training loops, etc.

    Args:
        args: arguments passed from the cli
        data_dir (str): path to input folder
        output_dir (str): path to output folder
        hyper_params (dict): hyper parameters from the config file
        mlf_logger (obj): MLFlow logger callback.
    """
    # __TODO__ change the hparam that are used from the training algorithm
    # (and NOT the model - these will be specified in the model itself)
    logger.info('List of hyper-parameters:')
    check_and_log_hp(
        ['architecture', 'batch_size', 'exp_name', 'max_epoch', 'optimizer', 'patience', 'seed'],
        hyper_params)

    if hyper_params["seed"] is not None:
        set_seed(hyper_params["seed"])

    log_exp_details(os.path.realpath(__file__), args)

    if not data_dir.endswith(".hdf5"):
        data_dir = os.path.join(data_dir, "extract_s5_raw.hdf5")
    if args.data_module=="hdf5":
        dm = selfsupmotion.data.objectron.hdf5_parser.ObjectronFramePairDataModule(
            hdf5_path=data_dir,
            input_height=hyper_params.get("input_height", 224),
            gaussian_blur=hyper_params.get("gaussian_blur", True),
            jitter_strength=hyper_params.get("jitter_strength", 1.0),
            batch_size=hyper_params["batch_size"],
            num_workers=hyper_params["num_workers"],
        )
    elif args.data_module=="file":
        dm = selfsupmotion.data.objectron.file_datamodule.ObjectronFileDataModule(num_workers=hyper_params["num_workers"],batch_size=hyper_params["batch_size"] )
        dm.setup() #In order to have the sample count.
    else:
        raise ValueError(f"Invalid datamodule specified on CLU : {args.data_module}")
    if "num_samples" not in hyper_params:
        # the model impl uses the sample count to prepare scheduled LR values in advance
        hyper_params["num_samples"] = len(dm.train_dataset) #dm.train_sample_count

    if "num_samples_valid" not in hyper_params:
        hyper_params["num_samples_valid"] = len(dm.val_dataset)

    model = load_model(hyper_params)

    train(model=model, optimizer=None, loss_fun=None, datamodule=dm,
          patience=hyper_params['patience'], output=output_dir,
          max_epoch=hyper_params['max_epoch'], use_progress_bar=not args.disable_progressbar,
          start_from_scratch=args.start_from_scratch, mlf_logger=mlf_logger)


if __name__ == '__main__':
    main()
