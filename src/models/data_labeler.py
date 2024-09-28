"""
Classes for performing dataset labeling

Raises:
    ValueError: "Empty diagnosis"

"""
import json
from collections import Counter

import numpy as np
import pandas as pd


class LabelClinicalData:
    """
    Label the data according to a majority vote strategy
    """

    def __init__(self,
                 clinical_map: str):
        with open(clinical_map, "r", encoding="utf8") as cmap:
            self.clinical_map = json.load(cmap)

    @staticmethod
    def _normalize_subject(diagnosis):
        if not diagnosis:
            raise ValueError('Empty diagnosis.')
        normalized = np.copy(diagnosis)
        for i in range(len(diagnosis) - 1):
            if diagnosis[i]:
                window = diagnosis[i:len(diagnosis)]
                majority = Counter(window)
                vote = majority[True] > majority[False]
                normalized[i:len(diagnosis)] = np.full((len(diagnosis) - i), vote)

        return list(normalized)

    def __call__(self, clinical_mri, *args, **kwargs) -> pd.DataFrame:
        diag_cols = ["dx1", "dx2", "dx3", "dx4", "dx5"]

        clinical_mri["Label"] = \
            clinical_mri.apply(
                lambda x:
                any(self.clinical_map[x[c]] for c in diag_cols), axis=1)

        subjects = clinical_mri.groupby("Subject")
        for k in subjects.groups.keys():
            subject = subjects.get_group(k)
            diagnosis = subject["Label"].to_list()
            indexes = list(subject.index)
            normalized = self._normalize_subject(diagnosis)
            if diagnosis != normalized:
                for i, value in enumerate(indexes):
                    clinical_mri.loc[value, "Label"] = normalized[i]

        return clinical_mri.sort_values("ADRC_ADRCCLINICALDATA_ID").reset_index(drop=True)

    def __str__(self):
        return self.__class__.__name__
