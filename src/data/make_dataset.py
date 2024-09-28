"""
This module is used for processing the raw MRI session into an usable dataset
"""
import glob
import os
import re

from pathlib import Path
from os.path import join
import logging
import click

import numpy as np
import pandas as pd
from dotenv import find_dotenv, load_dotenv
from p_tqdm.p_tqdm import p_map
from utils import MRIProcessor
from clinical import make_association


def _mri_scans_check(mr_dir: str, mr_list: list) -> list:
    """
    Checks if the required MR scans are available
    Args:
        mr_dir: directory where MR are stored
        mr_list: list of required scans

    Returns: the list of missing MR scans

    """
    available_scans = glob.glob(mr_dir + "**/anat*/*T1w*.nii.gz", recursive=True)
    available_ids = [re.match(r".*(OAS3\d{4}_MR_d\d{4}).*", x).groups()[0] for x in available_scans]
    available_ids = set(available_ids)
    return list(set(mr_list).difference(available_ids))


def _freesurfer_check(free_dir: str, freesurfer_list: list) -> list:
    """
    Checks if the required Freesurfers file are available
    Args:
        free_dir: directory where the freesurfer files are
        freesurfer_list: list of the required freesurfers

    Returns: missing freesurfer list

    """
    available_scans = glob.glob(free_dir + "**/mri/brainmask.mgz", recursive=True)
    available_ids = [re.match(r".*(OAS3\d{4}_MR_d\d{4}).*", x).groups()[0] for x in available_scans]
    available_ids = set(available_ids)
    result = []
    for freesurf in freesurfer_list:
        query = re.sub(r"Freesurfer5\d", "MR", freesurf)
        if query not in available_ids:
            result.append(freesurf)
    return result


def _find_mri_and_mask(mr_dir: str, free_dir: str, clinical: pd.DataFrame) -> pd.DataFrame:
    """
    Find all MRIs and corresponding Brainmasks in mr_dir and free_dir respectively
    Args:
        mr_dir: directory where there are the MRI scans
        free_dir: directory where there are Freesurfer files
        clinical: Dataframe with MR sessions associated to clinical sessions

    Returns: a copy of clinical Dataframe with 'mr_path', 'free_path' and 'check_passed' columns

    """
    clinical_mri = clinical.copy()
    for i, row in clinical_mri.iterrows():
        mr_id = row.MR_ID
        mri_list = glob.glob(join(mr_dir, mr_id, "anat*/*.nii.gz"))
        t1w_list = [p for p in mri_list if "T1w" in p]
        if len(t1w_list) > 0:
            t1w = list(sorted(t1w_list))[0]
            brainmask = glob.glob(join(free_dir, mr_id, "mri/brainmask.mgz"))
            if len(brainmask) > 0:
                clinical_mri.loc[i, 'mr_path'] = t1w
                clinical_mri.loc[i, 'free_path'] = brainmask[0]
                clinical_mri.loc[i, "check_passed"] = True
            else:
                clinical_mri.loc[i, "check_passed"] = False
        else:
            clinical_mri.loc[i, "check_passed"] = False
    clinical_mri.check_passed = clinical_mri.check_passed.fillna("False")
    return clinical_mri


@click.command()
@click.argument('source_mr', type=click.Path(exists=True))
@click.argument('source_freesurf', type=click.Path(exists=True))
@click.argument('output_directory', type=click.Path())
@click.argument('clinical_data', type=click.Path(exists=True))
@click.argument('mr_sessions', type=click.Path(exists=True))
@click.argument('ignore_missing', type=click.BOOL, required=False, default=True)
def main(source_mr: str,
         source_freesurf: str,
         output_directory: str,
         clinical_data: str,
         mr_sessions: str,
         ignore_missing: bool = True):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """

    dirname = os.path.dirname(__file__)
    ds_name = "data"
    os.makedirs(join(output_directory, ds_name), exist_ok=True)

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    logger.info('making final data set from raw data into %s', output_directory)

    clinical_mr = make_association(pd.read_csv(clinical_data), pd.read_csv(mr_sessions))
    logger.info('mr sessions have been associated with clinical assessments, (num entries: %s)', len(clinical_mr))

    clinical_mr.to_csv(join(output_directory, "clinical-mr-full.csv"), index=False)
    logger.info('clinical association saved to %s', join(output_directory, "clinical-mr.csv"))

    process_list = clinical_mr[clinical_mr.MR_ID.notna() & clinical_mr.Freesurfer_ID.notna()].reset_index(
        drop=True).copy()

    missing_mr = _mri_scans_check(source_mr, process_list.MR_ID)
    missing_freesurf = _freesurfer_check(source_freesurf, process_list.Freesurfer_ID)
    if len(missing_mr) > 0 or len(missing_freesurf) > 0:
        logger.info('there are some missing files, dump missing into interim folder')

        pd.DataFrame(missing_mr).to_csv(join(dirname, "../../data/interim/missing-scans.csv"),
                                        index=False,
                                        header=False)
        pd.DataFrame(missing_freesurf).to_csv(join(dirname, "../../data/interim/missing-freesurfer.csv"),
                                              index=False)
        logger.debug('missing scans: %s', missing_mr)
        logger.debug('missing freesurfer %s', missing_freesurf)
        if not ignore_missing:
            logger.error('There are missing scans and cannot be downloaded cannot continue')
            raise FileNotFoundError('There are missing files and ignore is set to false')

    process_list = clinical_mr[clinical_mr.MR_ID.notna() & clinical_mr.Freesurfer_ID.notna()] \
        .reset_index(drop=True).copy()

    process_list.to_csv(join(output_directory, "clinical-mr.csv"), index=True)
    logger.info('selected %s that have both MR session and freesurfer entries', len(process_list))

    mri_processor = MRIProcessor()

    logger.info('scan files in search mri sessions without corresponding freesurfer')
    sessions = _find_mri_and_mask(source_mr, source_freesurf, process_list)
    not_conforms = sessions[~sessions.check_passed]
    logger.info('found %s MRIs without freeesurfer, dump info to file invalid-sessions.csv', len(not_conforms))
    not_conforms.to_csv(join(output_directory, "invalid-sessions.csv"), index=False)
    sessions = sessions[sessions.check_passed].reset_index(drop=True)

    def save_mr(row):
        _, row = row
        mri, min_val, max_val = mri_processor(row.mr_path, row.free_path)
        np.savez_compressed(join(output_directory, ds_name, row.MR_ID), x=mri)
        row.loc['min_value'] = min_val
        row.loc['max_value'] = max_val
        return row

    logger.info("start processing mr sessions")
    results = p_map(save_mr, sessions.iterrows(), desc="Processing MRIs", total=len(sessions))

    logger.info("processing ended")
    sessions = pd.DataFrame(results)
    sessions.drop(columns=["mr_path", "free_path", "check_passed", "Freesurfer_ID"], inplace=True)

    logger.info("save index file")
    sessions.to_csv(join(output_directory, ds_name, "index.csv"), index=False)

    logger.info("Done!")


if __name__ == '__main__':
    LOG_FMT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=LOG_FMT)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()  # pylint: disable=no-value-for-parameter
