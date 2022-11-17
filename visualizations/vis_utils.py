#!python
# -*- coding: utf-8 -*-
"""
Utils used in visualizations, like pretty printing,
and specific data structure manipulation.
"""


def get_var_filter_iter(df, var=None):
    """
    Returns number of unique values of var in df and iterator going over masks
    selecting for each value. If var is None only one full mask is yielded.
    :return: len(vals), Yield[val, df[var] == val]
    """
    if var is None:
        vals = [""]
    else:
        vals = df[var].unique()

    def val_iterator():
        if var is None:
            yield "", np.ones(len(df), dtype=bool)
        else:
            for val in vals:
                yield val, df[var] == val

    return len(vals), val_iterator


def fill_dict(keys, vals):
    """
    Creates dictionary with key value pairs where val is not None..
    """
    return {key: val for key, val in zip(keys, vals) if val is not None}


def none2str(v):
    """
    str() wrapper, if v is None returns empty string.
    """
    if v is None:
        return ""
    return str(v)