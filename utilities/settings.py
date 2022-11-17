#!python
# -*- coding: utf-8 -*-
"""
Settings and utils for managing settings saved in CSV.
"""

__author__ = "Tomas Zitka"
__email__ = "zitkat@kky.zcu.cz"

from pathlib import Path
import pandas as pd
import numpy as np
import csv


def load_settings(set_path: Path = Path("rendering_settings.csv"),
                  set_name: str = "Default",
                  dtype_col: str = "Dtype",
                  group_sep=":Group:"):
    """
    Loads settings from csv or ods file table, mandatory columns are
    Key, Default, any other columns are treated as versions of the same variables.
    :param set_path: path to the settings file
    :param set_name: name of the column to load
    :param group_sep: string used to mark start of different groups,
                      should be in Key column, with name of the group in Default,
                      variables that are not defined inside any group are put directly
                      into output dictionary
    :return: dictionary containing dictionaries for each group of settings and
             key:value pairs for ungrouped variables
    """
    if set_path.suffix == ".csv":
        settings_df = pd.read_csv(set_path, sep=",", index_col="Key", comment="#",
                                  quoting=csv.QUOTE_NONE)
    elif set_path.suffix == ".ods":
        settings_df = pd.read_excel(set_path, engine='odf', index_col="Key")
    else:
        raise NotImplemented("Only csv or ods files supported.")

    if set_path.stem[-1] == "T":
        settings_df = settings_df.T
        settings_df.index = [l.split(".")[0] if group_sep in l else l for l in settings_df.index]

    splits = list(np.where(settings_df.index == group_sep)[0])
    if len(splits) == 0 or splits[0] != 0:
        splits = [0] + splits
    splits += [len(settings_df)]
    starts_ends = list(zip(splits[:-1], splits[1:]))

    all_settings = {}
    for start, end in starts_ends:
        if settings_df.iloc[start].name == group_sep:
            settings_name = settings_df.iloc[start]["Default"].strip()
            # skip row with group name
            start += 1
        else:
            settings_name = None
        raw_settings = settings_df.iloc[start : end][set_name].to_dict()
        dtypes = settings_df.iloc[start : end][dtype_col].to_dict()

        curr_settings = {}
        for k, v in raw_settings.items():
            if v == "None":
                curr_settings[k] = None
            elif isinstance(v, str) and dtypes[k] != "str":
                curr_settings[k] = eval(v, globals(), curr_settings)
            else:
                curr_settings[k] = v
        if settings_name:
            all_settings[settings_name] = curr_settings
        else:
            all_settings.update(curr_settings)

    return all_settings
