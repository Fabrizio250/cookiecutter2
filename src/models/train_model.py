"""
This module is used for training the model
"""
from argparse import ArgumentParser
import logging
import os.path
from pathlib import Path
import multiprocessing

import mlflow.pytorch
import pandas as pd
import pytorch_lightning as pl
import torch
from dotenv import load_dotenv, find_dotenv
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from src.models.mri_classifier import MRIClassifier
from src.models.data_loader import MROasis3Datamodule
from src.models.data_labeler import LabelClinicalData
from src.models.data_splitter import HoldoutSplitter
from src.models.utils.balancers import RandomDownsampler


def auto_accelerator():
    """
    If no accelerator is provided this method select the GPU in case is available
    Returns:
        'cpu' if no available GPU is found otherwise return 'gpu'
    """
    logging.info('--accelerator not specified auto selection')
    if torch.cuda.is_available():
        accelerator = 'gpu'
        logging.info('GPU found will use it')
    else:
        accelerator = 'cpu'
        logging.info('No available GPU will use CPU')

    return accelerator


def check_precision(accelerator, precision):
    """
    Check if the 16 bit precision is permitted,
     16 bit precision can be used only if a GPU is
     both available and enabled
    Args:
        accelerator: the selected accelerator

    Returns: 16 if GPU available and enabled otherwise 32

    """
    if precision == 16:
        logging.info('--precision parameter set to 16, check if available')
        if torch.cuda.is_available() and accelerator.lower() == 'gpu':
            logging.info('GPU is available and enabled 16bit ok for training speedup')
            precision = 16
        else:
            logging.info('GPU not available or not enabled will use 32bit precision')
            precision = 32
    return precision


def auto_workers(accelerator):
    """
    Detect the maximum number of cpu workers and select
    how many use for data loading based on the accelerator
    if training is on GPU will use all available processors
    otherwise only one
    Args:
        accelerator: the selected accelerator cpu/gpu

    Returns: the number of workers to use for data loading

    """
    logging.info('number of workers not specified auto select')
    cpu_count = multiprocessing.cpu_count()
    logging.info('Maximum available workers %s', cpu_count)
    if accelerator.lower() == 'gpu' and torch.cuda.is_available():
        logging.info('Train on GPU enabled use all CPU cores for workers')
        workers = cpu_count
    else:
        logging.info('Train on CPU will use 1 core')
        workers = 1
    return workers


def prepare_data(clinical_dict, clinical_data, data_root, random_seed=None):
    """
    Prepare the dataset to be used in training, the preparation consists
    in labeling the data apply downsampling and feed into a BaseSplitter
    Args:
        clinical_dict: path to json dictionary of diseases
        clinical_data: path to clinical data with sessions
        data_root: root folder of the preprocessed MR
        random_seed: seed for the splitter

    Returns: the base splitter

    """
    clinical_mr = pd.read_csv(clinical_data)
    labeler = LabelClinicalData(clinical_dict)
    clinical_mr = labeler(clinical_mr)
    index = pd.merge(pd.read_csv(os.path.join(data_root, 'index.csv')),
                     clinical_mr[['MR_ID', "Label"]],
                     on="MR_ID",
                     how='left')
    sampler = RandomDownsampler(index, random_state=random_seed)
    clinical_mr = sampler.get_sample()
    splitter = HoldoutSplitter(clinical_mr)

    return splitter


if __name__ == '__main__':
    LOG_FMT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=LOG_FMT)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())
    parser = ArgumentParser()
    parser.add_argument('--seed', type=int, required=False)
    parser.add_argument('--workers', type=int, required=False)
    parser.add_argument('--batch_size', type=int, required=True)
    parser.add_argument('--diagnosis_dict', type=str, required=True,
                        help='path to the json used as dictionary ')
    parser.add_argument('--clinical_data', type=str, required=True,
                        help='path to csv file containing clinical data')
    parser.add_argument('--data_folder', type=str, required=True,
                        help='path to folder where the preprocessed MRI are stored')
    parser.add_argument('--mlflow_uri', type=str, required=False, default=None,
                        help='url to mlfow server'
                        )
    parser = MRIClassifier.add_model_specific_args(parser)
    parser = pl.Trainer.add_argparse_args(parser)

    args = parser.parse_args()
    if args.accelerator is None:
        args.accelerator = auto_accelerator()

    args.precision = check_precision(args.accelerator, args.precision)

    if args.workers is None:
        args.workers = auto_workers(args.accelerator)

    if args.seed is not None:
        logging.info("random seed provided use it")
        s = pl.seed_everything(args.seed, workers=True)
        logging.info("seed everithing with %d", s)

    data_splitter = prepare_data(args.diagnosis_dict,
                                 args.clinical_data,
                                 args.data_folder, args.seed)

    data_module = MROasis3Datamodule(args.data_folder,
                                     data_splitter,
                                     batch_size=args.batch_size, num_workers=args.workers)

    torch.cuda.empty_cache()
    checkpoints = ModelCheckpoint(monitor='val_loss',
                                  filename='mri_classifier-{epoch:02d}-{val_loss:.2f}',
                                  save_top_k=2,
                                  mode='min',
                                  dirpath=os.path.join(project_dir, "models/", "checkpoints/"))
    early_stop = EarlyStopping(patience=17, monitor="val_loss", mode='min', min_delta=5e-3)

    vargs = vars(args)
    model = MRIClassifier(**vargs)

    trainer = pl.Trainer.from_argparse_args(args, callbacks=[early_stop, checkpoints])

    if args.mlflow_uri is not None:
        mlflow.set_tracking_uri(args.mlflow_uri)
    mlflow.pytorch.autolog()

    additional_params = {'random_seed': args.seed,
                         'resample_strategy': RandomDownsampler.__name__,
                         'mini_batch_size': args.batch_size,
                         'accumulate_grad_batches': args.accumulate_grad_batches}

    with mlflow.start_run() as r:
        mlflow.log_params(additional_params)
        trainer.fit(model, datamodule=data_module)
        trainer.test(ckpt_path='best', datamodule=data_module)

    trainer.save_checkpoint(filepath=os.path.join(project_dir, 'models/', "model.ckpt"))
