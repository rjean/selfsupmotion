
import glob
import logging
import os

import mlflow
import orion
import yaml
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from orion.client import report_results
from yaml import dump
from yaml import load

logger = logging.getLogger(__name__)

BEST_MODEL_NAME = 'best_model'
LAST_MODEL_NAME = 'last_model'
STAT_FILE_NAME = 'stats.yaml'


def write_stats(output, best_eval_score, epoch, remaining_patience):
    """Write statistics of the best model at the end of every epoch.

    Args:
        output (str): Output directory
        best_eval_score (float): best score obtained on evaluation set.
        epoch (int): Which epoch training is at.
        remaining_patience (int): How many more epochs before training stops.
    """
    mlflow_run = mlflow.active_run()
    mlflow_run_id = mlflow_run.info.run_id if mlflow_run is not None else 'NO_MLFLOW'
    to_store = {'best_dev_metric': best_eval_score, 'epoch': epoch,
                'remaining_patience': remaining_patience,
                'mlflow_run_id': mlflow_run_id}
    with open(os.path.join(output, STAT_FILE_NAME), 'w') as stream:
        dump(to_store, stream)


def load_stats(output):
    """Load the latest statistics.

    Args:
        output (str): Output directory
    """
    with open(os.path.join(output, STAT_FILE_NAME), 'r') as stream:
        stats = load(stream, Loader=yaml.FullLoader)
    return stats['best_dev_metric'], stats['epoch'], stats['remaining_patience'], \
        stats['mlflow_run_id']


def train(**kwargs):  # pragma: no cover
    """Training loop wrapper. Used to catch exception if Orion is being used."""
    best_dev_metric = train_impl(**kwargs)
    try:
        pass
        best_dev_metric = train_impl(**kwargs)
    except RuntimeError as err:
        if orion.client.IS_ORION_ON and 'CUDA out of memory' in str(err):
            logger.error(err)
            logger.error('model was out of memory - assigning a bad score to tell Orion to avoid'
                         'too big model')
            best_dev_metric = -999
        else:
            raise err

    report_results([dict(
        name='dev_metric',
        type='objective',
        # note the minus - cause orion is always trying to minimize (cit. from the guide)
        value=-float(best_dev_metric))])


def reload_model(output, model_name, model, optimizer,
                 start_from_scratch=False):  # pragma: no cover
    """Reload a model.

    Can be useful for model checkpointing, hyper-parameter optimization, etc.

    Args:
        output (str): Output directory.
        model_name (str): Name of the saved model.
        model (obj): A model object.
        optimizer (obj): Optimizer used during training.
        start_from_scratch (bool): starts training from scratch even if a saved moel is present.
    """
    saved_model = os.path.join(output, model_name)
    if start_from_scratch and os.path.exists(saved_model):
        logger.info('saved model file "{}" already exists - but NOT loading it '
                    '(cause --start_from_scratch)'.format(output))
        return
    if os.path.exists(saved_model):
        logger.info('saved model file "{}" already exists - loading it'.format(output))

        model.load_state_dict(torch.load(saved_model))
    if os.path.exists(output):
        logger.info('saved model file not found')
        return

    logger.info('output folder not found')
    os.makedirs(output)


def train_impl(
        model,
        optimizer,
        loss_fun,
        datamodule,
        patience,
        output,
        max_epoch,
        use_progress_bar,
        start_from_scratch,
        mlf_logger,
        precision
):  # pragma: no cover
    """Main training loop implementation.

    Args:
        model (obj): The neural network model object.
        optimizer (obj): Optimizer used during training.
        loss_fun (obj): Loss function that will be optimized.
        datamodule (obj): DataModule that contains both train/valid data loaders.
        patience (int): max number of epochs without improving on `best_eval_score`.
            After this point, the train ends.
        output (str): Output directory.
        max_epoch (int): Max number of epochs to train for.
        use_progress_bar (bool): Use tqdm progress bar (can be disabled when logging).
        start_from_scratch (bool): Start training from scratch (ignore checkpoints)
        mlf_logger (obj): MLFlow logger callback.
    """
    best_checkpoint_callback = ModelCheckpoint(
        filepath=os.path.join(output, BEST_MODEL_NAME),
        save_top_k=1,
        verbose=use_progress_bar,
        monitor="val_loss",
        mode="auto",
        period=1
    )

    last_model_path = os.path.join(output, LAST_MODEL_NAME)
    last_checkpoint_callback = ModelCheckpoint(
        filepath=last_model_path,
        verbose=use_progress_bar,
        period=1
    )

    last_models = glob.glob(last_model_path + '*')

    if len(last_models) > 1:
        raise ValueError('more than one last model found to resume - provide only one')
    elif len(last_models) == 1:
        logger.info('resuming training from {}'.format(last_models[0]))
        resume_from_checkpoint = last_models[0]
    else:
        logger.info('starting training from scratch')
        resume_from_checkpoint = None

    early_stopping = EarlyStopping("val_accuracy", mode="auto", patience=patience,
                                   verbose=use_progress_bar)
    
    trainer = pl.Trainer(
        # @@@@@@@@@@@ TODO check if we can add an online evaluator w/ callback
        callbacks=[early_stopping, best_checkpoint_callback, last_checkpoint_callback],
        checkpoint_callback=True,
        logger=mlf_logger,
        max_epochs=max_epoch,
        resume_from_checkpoint=resume_from_checkpoint,
        gpus=torch.cuda.device_count(),
        auto_select_gpus=True,
        precision=precision,
        amp_level="O1",
        accelerator=None,  # @@@@@@@@@ TODO CHECK ME OUT w/ precision arg too
    )

    trainer.fit(model, datamodule=datamodule)
    best_dev_result = float(early_stopping.best_score.cpu().numpy())
    return best_dev_result
