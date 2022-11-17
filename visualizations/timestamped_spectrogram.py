#!python
# -*- coding: utf-8 -*-
"""
Tools for correctly plotting spectrogram with non-uniform time stamps and gaps
and optionally labeled time intervals.
"""


from typing import Tuple, Sequence, Any, TypeVar

import numpy as np
from numpy.typing import NDArray


import copy
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as ptchs
from datetime import datetime

SPECT_LEN = TypeVar("SPECT_LEN")


def create_time_labels(times: np.ndarray, end_time, step=5):
    if step == 0:
        step = 1
    times_series = pd.Series(times)
    time_labels = list(times_series[::step].dt.strftime("%H:%M")) \
                    + [times_series.iloc[-1].strftime("%H:%M")]
    if end_time is not None:
        time_labels += [end_time.astype(datetime).strftime("%H:%M")]
    else:
        time_labels += [""]
    return time_labels


def create_time_ticks(times: np.ndarray, end_time, step=5):
    if step == 0:
        step = 1
    ticks = np.zeros((len(times[::step]) + 2))
    ticks[:-2] = times[::step]
    ticks[-2] = times[-1]
    ticks[-1] = end_time
    return ticks


def plot_day_spectra(spectra: NDArray[(Any, SPECT_LEN), int],
                     temp: NDArray[float],
                     times: NDArray[np.datetime64],
                     hive: str = None,
                     true_labels: Sequence[Tuple[np.datetime64,
                                                 np.datetime64,
                                                 int]] = (),
                     predicted_labels: Sequence[Tuple[np.datetime64,
                                                      np.datetime64,
                                                      int]] = (),
                     end_time: np.datetime64 = None,
                     freq_bins: np.ndarray = None,  # TODO document parameter
                     fig: plt.Figure = None,
                     ax: plt.Axes = None) -> Tuple[plt.Figure, plt.Axes]:
    """
    # TODO generalize to arbitrary time period
    All datetime64 objects need to be in seconds.
    """

    day = np.datetime_as_string(times[0], "D")
    midnight = get_midnight(times[0]).astype(int)
    rel_times = times.astype(int) - midnight
    if end_time is not None:
        rel_end_time = end_time.astype(int) - midnight
    else:
        rel_end_time = 24*60*60  # last frame extends to the end of the day

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(10, 7))
        fig.canvas.set_window_title(f"{hive}: {day}")

    ax.text(.01, 1.02, f"Hive: {hive} \n"
                       f"Day:  {day}",
            transform=ax.transAxes)

    ax.set_xlim(0, 24*60*60)  # x axis is in second and spans one day
    cmap = copy.copy(mpl.cm.get_cmap("jet"))
    cmap.set_bad('grey')
    masked_spectra = np.ma.array(spectra.T, mask=spectra.T < 0)
    imsmin = 0
    imsmax = np.nanmax(masked_spectra)
    for i in np.arange(len(spectra) - 1):
        plt.imshow(masked_spectra[:, i][:, None], origin="lower", aspect="auto",
                   interpolation="none",
                   cmap=cmap, vmin=imsmin, vmax=imsmax,
                   extent=(rel_times[i], rel_times[i+1],  # x extent
                           freq_bins[0], freq_bins[-1]))  # y extent

    plt.imshow(masked_spectra[:, -1][:, None], origin="lower", aspect="auto",
               interpolation="none",
               cmap=cmap, vmin=imsmin, vmax=imsmax,
               extent=[rel_times[-1], rel_end_time,
                       freq_bins[0], freq_bins[-1]])

    ax.set_xlabel("Time [HH:MM]")
    ax.set_ylabel("Frequency [HZ]")
    ax.set_xticks(create_time_ticks(rel_times, end_time=rel_end_time,
                                    step=len(times) // 12))
    ax.set_xticklabels(create_time_labels(times, end_time=end_time,
                                          step=len(times) // 12),
                       rotation=45, ha="right")

    ax.set_yticks(freq_bins[::4])
    ax.set_yticklabels(freq_bins[::4])

    rax = ax.twinx()
    rax.plot((rel_times[1:] + rel_times[:-1]) / 2, temp[:-1], lw=2, color="w")
    rax.plot((rel_times[1:] + rel_times[:-1]) / 2, temp[:-1], lw=1, color="r")
    rax.set_ylabel("Temperature [°C]")
    rax.set_ylim(0, 40)

    # ax.text(-1, 1280, f"True", fontsize=6, horizontalalignment="right")
    # ax.text(-1, 1280, f"Pred", fontsize=6, horizontalalignment="right")

    ax.set_ylim(0, 1320)
    for offset, labels in zip([1280, 1300], [true_labels, predicted_labels]):
        for start, end, lab in labels:
            bh_col = bh_colors[lab] if lab in bh_colors else "grey"
            ax.add_patch(ptchs.Rectangle((start.astype(int) - midnight, offset),
                                         (end - start).astype(int),
                                         20,
                                         facecolor=bh_col,
                                         edgecolor='w'))

    ax.grid()

    return fig, ax
