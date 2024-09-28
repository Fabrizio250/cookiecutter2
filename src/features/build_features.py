import datetime
import glob
import os
from os.path import join
import logging
import click

import numpy as np
import pandas as pd
import tqdm
from dotenv import find_dotenv, load_dotenv

from app.backend.task.model import MRIClassifier
from skimage.util import random_noise

@click.command()
@click.argument('processed_directory', type=click.Path(exists=True))
@click.argument('output_directory', type=click.Path())
@click.argument('model_path', type=click.Path(exists=True))
@click.argument('add_noise', type=click.BOOL, default=False)
def main(processed_directory: str,
         output_directory: str,
         model_path: str,
         add_noise: bool = False):
    """
    Executes the feature extraction part of the model and save the results into a numpy array
    """

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    logger.info('starting feature extraction from processed data from %s', processed_directory)
    files = glob.glob(processed_directory + "/*.npz")
    logger.info('found %d files into %s', len(files), processed_directory)

    model = MRIClassifier.load_from_checkpoint(model_path)
    model.freeze()
    logger.info("model loaded from %s", model_path)
    timestamp = str(datetime.datetime.now().timestamp())
    result = np.zeros((len(files), 9216))
    for i, file in tqdm.tqdm(enumerate(files), desc="Extracting features", total=len(files)):
        sample = np.load(files[i])["x"]
        if add_noise:
            sample = random_noise(sample, mode='gaussian')
        _, feat = model.predict(sample)
        result[i, :] = feat

    np.savez_compressed(join(output_directory, f"features_{timestamp}.npz"),
                        data=result,
                        timestamps=[timestamp for _ in range(len(files))])


if __name__ == "__main__":
    LOG_FMT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=LOG_FMT)

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()  # pylint: disable=no-value-for-parameter
