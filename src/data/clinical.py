"""
This module provide methods for associate the MR sessions with clinical data
"""
from typing import Literal

import numpy as np
import pandas as pd
from pandas import DataFrame

_columns_ordered = ['ADRC_ADRCCLINICALDATA_ID', 'Subject', 'MR_ID',
                    'ageAtEntry', 'Age',
                    'DaysAfterEntry_clinic', 'DaysAfterEntry_mr', 'DaysDistance',
                    'Scanner', 'T1w', 'T2w', 'Freesurfer_ID',
                    'mmse','cdr',
                    'dx1', 'dx2', 'dx3', 'dx4', 'dx5',
                    'commun','homehobb', 'judgment', 'memory', 'orient',
                    'perscare', 'sumbox', 'apoe', 'height', 'weight']

_clinical_required_columns = ['ADRC_ADRCCLINICALDATA_ID', 'Subject', 'mmse',
                              'cdr', 'dx1', 'dx2', 'dx3', 'dx4', 'dx5']

_mr_required_columns = ['MR_ID', 'DaysAfterEntry',
                        'Scanner', 'Freesurfer_ID']


def make_association(clinical_data: DataFrame,
                     mri_sessions: DataFrame,
                     direction: Literal['nearest', 'forward', 'backward'] = 'nearest',
                     return_complete: bool = True) -> DataFrame:
    """
    Associate MR sessions list with corresponding clinical assessment,
    the "correspondence" is driven by 'direction' parameter,
    nearest means the closest in time between session and clinical assessment;
    forward the mr session is associated with the first subsequent clinical assessment,
    backward the mr session is associated with the previous clinical assessment
    Args:
        clinical_data: dataframe of clinincal assessment
            must follow the structure defined in the dataset card
        mri_sessions: dataframe of mri sessions,
            must follow the structure defined in the dataset card
        direction: one between nearest, backward or forward
        return_complete: if True this method return also the clinical assessment
            without an associated MR session, otherwise will return
            only the clinical assessments with a mr session

    Returns: the Dataframe of clinical assessments and mr sessions,
     fields described in the dataset card

    """

    assert set(_clinical_required_columns).issubset(clinical_data.columns), \
        f"Provided clinical data does not have required fields, missing fields are: " \
        f"{set(_clinical_required_columns).difference(clinical_data.columns)}"

    assert set(_mr_required_columns).issubset(mri_sessions.columns), \
        f"Provided mr sessions data does not have required fields, missing fields are: " \
        f"{set(_mr_required_columns).difference(mri_sessions.columns)} "

    clinical_data["DaysAfterEntry_clinic"] = clinical_data["ADRC_ADRCCLINICALDATA_ID"].apply(
        lambda x: int(x.split("_")[-1][1:]))
    mri_sessions.rename(columns={'DaysAfterEntry': 'DaysAfterEntry_mr'}, inplace=True)

    sub_no_clinical = set(mri_sessions.Subject).difference(clinical_data.Subject)
    mri_sessions = mri_sessions[~mri_sessions.Subject.isin(sub_no_clinical)]\
        .reset_index(drop=True).copy()

    mr_clinical = pd.merge_asof(mri_sessions.sort_values("DaysAfterEntry_mr"),
                                clinical_data.sort_values("DaysAfterEntry_clinic"),
                                right_on='DaysAfterEntry_clinic',
                                left_on='DaysAfterEntry_mr', by='Subject',
                                direction=direction)

    mr_clinical["DaysDistance"] = \
        np.abs(mr_clinical.DaysAfterEntry_clinic - mr_clinical.DaysAfterEntry_mr)

    mr_clinical = mr_clinical[mr_clinical.ADRC_ADRCCLINICALDATA_ID.notna()]

    if return_complete:
        mr_clinical = pd.concat([mr_clinical, clinical_data]).sort_values("MR_ID").drop_duplicates(
            subset="ADRC_ADRCCLINICALDATA_ID")

    mr_clinical = mr_clinical.sort_values("ADRC_ADRCCLINICALDATA_ID").reset_index(drop=True)
    mr_clinical = mr_clinical[_columns_ordered]

    return mr_clinical
