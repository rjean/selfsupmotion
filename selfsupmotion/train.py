
import glob
import logging
import os

import torch
import pl_bolts
import pytorch_lightning as pl
from orion.client import report_results

logger = logging.getLogger(__name__)

BEST_MODEL_NAME = 'best_model'
LAST_MODEL_NAME = 'last_model'
STAT_FILE_NAME = 'stats.yaml'


def train(**kwargs):  # pragma: no cover
    """Training loop wrapper. Used to catch exception if Orion is being used."""
    best_dev_metric = train_impl(**kwargs)
    try:
        # TODO @@@@@@@@@@@@@@@@@@@@@@ FIXME
        pass
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
        tbx_logger,
        precision,
        early_stop_metric,
        accumulate_grad_batches
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
        tbx_logger (obj): TensorBoard logger callback.
    """
    best_checkpoint_callback = pl.callbacks.ModelCheckpoint(
        filepath=os.path.join(output, BEST_MODEL_NAME),
        save_top_k=1,
        verbose=use_progress_bar,
        monitor="val_loss",
        mode="auto",
        period=1
    )

    last_model_path = os.path.join(output, LAST_MODEL_NAME)
    last_checkpoint_callback = pl.callbacks.ModelCheckpoint(
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

    if early_stop_metric != "none":
        early_stopping = pl.callbacks.EarlyStopping(early_stop_metric, mode="auto", patience=patience, verbose=use_progress_bar)
        callbacks = [early_stopping]
    else:
        callbacks = []

    printer_callback = pl_bolts.callbacks.PrintTableMetricsCallback()
    callbacks = callbacks.extend([
        best_checkpoint_callback,
        last_checkpoint_callback,
        printer_callback,
    ])
    trainer = pl.Trainer(
        # @@@@@@@@@@@ TODO check if we can add an online evaluator w/ callback
        callbacks=callbacks,
        checkpoint_callback=True,
        logger=[mlf_logger, tbx_logger],
        max_epochs=max_epoch,
        resume_from_checkpoint=resume_from_checkpoint,
        gpus=torch.cuda.device_count(),
        auto_select_gpus=True,
        precision=precision,
        amp_level="O1",
        accelerator=None,
        accumulate_grad_batches=accumulate_grad_batches
    )

    trainer.fit(model, datamodule=datamodule)
    best_dev_result = float(early_stopping.best_score.cpu().numpy())
    return best_dev_result
