#!python
# -*- coding: utf-8 -*-

__author__ = "Tomas Zitka"
__email__ = "zitkat@kky.zcu.cz"

import re
import time
from pathlib import Path
from typing import Tuple, Dict, List, Any

import unicodedata
from PIL.Image import Image as PILImage

import pandas as pd

import click
import ast
from functools import singledispatch, update_wrapper, lru_cache, wraps

from PIL import Image


def make_path(*pathargs, isdir=False, **pathkwargs):
    new_path = Path(*pathargs, **pathkwargs)
    return ensured_path(new_path, isdir=isdir)


def ensured_path(path: Path, isdir=False):
    if isdir:
        path.mkdir(parents=True, exist_ok=True)
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
    return path


def now():
    """
    :return: date and time as YYYY-mm-dd-hh-MM
    """
    return time.strftime("%Y-%m-%d-%H-%M")


class PythonLiteralOption(click.Option):

    def type_cast_value(self, ctx, value):
        try:
            return ast.literal_eval(value)
        except:
            raise click.BadParameter(value)


def methodispatch(func):
    dispatcher = singledispatch(func)

    def wrapper(*args, **kw):
        return dispatcher.dispatch(args[1].__class__)(*args, **kw)

    wrapper.register = dispatcher.register
    update_wrapper(wrapper, func)
    return wrapper


def load_img_rgb_withrotation(file_path: Path) -> PILImage:
    """
    Correctly loads image from file using Pillow:
        - rotates according to exif tag 274
        - forces RGB conversion
    Args:
        file_path: path to image

    Returns:
        Pillow Image object, without file metadata due to conversion.
    """
    in_img = Image.open(str(file_path))
    exif = in_img.getexif()
    if exif is not None:
        orientation_tag = 274
        # from PIL.ExifTags.TAGS
        # [(k, v) for k, v in  PIL.ExifTags.TAGS.items() if v in ["Orientation"]
        if exif.get(orientation_tag, None) == 3:
            in_img = in_img.rotate(180, expand=True)
        elif exif.get(orientation_tag, None) == 6:
            in_img = in_img.rotate(270, expand=True)
        elif exif.get(orientation_tag, None) == 8:
            in_img = in_img.rotate(90, expand=True)

    return in_img.convert('RGB')


def slugify(value, allow_unicode=False):
    """
    Taken from https://github.com/django/django/blob/master/django/utils/text.py
    Convert to ASCII if 'allow_unicode' is False. Convert spaces or repeated
    dashes to single dashes. Remove characters that aren't alphanumerics,
    underscores, or hyphens. Convert to lowercase. Also strip leading and
    trailing whitespace, dashes, and underscores.
    """
    value = str(value)
    if allow_unicode:
        value = unicodedata.normalize('NFKC', value)
    else:
        value = unicodedata.normalize('NFKD', value).encode('ascii', 'ignore').decode('ascii')
    value = re.sub(r'[^\w\s-]', '', value)
    return re.sub(r'[-\s]+', '-', value).strip('-_')


def report(*args, end="\n"):
    """Print and return arguments.

    Might not always behave most intuitively due to difficulties of creating
     general identity  for variadic function.

    :param args: args to print and return
    :param end: end parameter for print
    :return:
    """
    print(*args, end=end)
    return args[0] if len(args) == 1 else args


def tupleasdict(tup, keys=None) -> Dict[Any, Any]:
    if keys is None:
        keys = range(len(tup))
    if len(keys) != len(tup):
        raise ValueError("Number of keys must be the same as tuple length")
    return {k: t for k, t in zip(keys, tup)}


def f_and(*fargs):
    return lambda *args, **kwargs: all(f(*args, **kwargs) for f in fargs)


def f_or(*fargs):
    return lambda *args, **kwargs: any(f(*args, **kwargs) for f in fargs)


def partial_keywords(f, **fkwargs):
    return lambda *args, **kwargs: f(*args, **{**fkwargs, **kwargs})


def join_tail(n: int, seq: List[str]):
    return seq[:n] + [" ".join(seq[n:])]


def df_file_cache(filename: str, data_folder: str):
    """
    Decorates function of type
        *str -> pandasDataFrame
    to cache the output dataframe under provided filename
    in data_folder. If the decorated method receives any arguments they must
    be strings. These are joined and _prefixed_ to the filename.
    :param filename:
    :param data_folder:
    :return:
    """

    file_path = Path(filename)
    file_path_in_dataf = Path(data_folder) / file_path

    def file_cache_dec(mth):
        @wraps(mth)
        def file_cache_wrapper(self, *args, **kwargs):

            prefix = "_".join(args) + "_".join(kwargs.values())

            full_file_path_in_dataf = file_path_in_dataf.with_name(
                    prefix + file_path_in_dataf.name)

            if full_file_path_in_dataf.exists():
                df = pd.read_csv(full_file_path_in_dataf, index_col=0)
                for col in df.columns:
                    if "time" in col.lower():
                        df[col] = pd.to_datetime(df[col])
                return df
            else:
                df = mth(self, *args, **kwargs)
                df.to_csv(ensured_path(full_file_path_in_dataf))
                return df

        return file_cache_wrapper

    return file_cache_dec


def try_wait(tries=3, to_except=TimeoutError, delay: int = 3):

    def try_wait_dec(mth):

        @wraps(mth)
        def try_wait_wrapper(*args, **kwargs):
            for tr in range(tries):
                try:
                    res = mth(*args, **kwargs)
                    return res
                except to_except as e:
                    if tr == tries - 1:
                        raise e
                    time.sleep(delay)
        return try_wait_wrapper

    return try_wait_dec


def get_seq_constructor(guide):
    """Create constructor for given sequence type.

    Treats strings - the constructor is "".join
    """
    return "".join if isinstance(guide, str) else type(guide)


def collapse_sep(guide, *args, sep=" "):
    """Collapses all elements in sequences that correspond to sep in guide to the
    first such element.

    E.g. for  guide "Hello     world"
              args [[1, 1, 1, 1 ,1, 0, 0.1, 0.2, 0.3, 0.4, 2, 2, 2, 2, 2]]
              sep = " "
         returns [[1, 1, 1, 1, 1, 0 , 2, 2, 2, 2, 2]]

    :param guide: sequence
    :param args: must be sequences
    :param sep: possible element of guide
    :return: collapsed guide, *collapsed sequences,
    """
    if not all(len(guide) == len(s) for s in args):
        raise ValueError("All sequences must be the same length")

    res = []
    counter = 0
    for ii, item in enumerate(guide):
        if item == sep:
            counter += 1
        else:
            counter = 0

        if counter > 1:
            continue
        else:
            res.append((item, *(s[ii] for s in args)))

    return map(lambda inp, out: get_seq_constructor(inp)(out), [guide, *args], zip(*res))
