"""
Copyright (c) 2025, Spanish National Research Council (CSIC)
Developed by the Barcelona Center for Subsurface Imaging (B-CSI)

This source code is subject to the terms of the
GNU Lesser General Public License.
"""

from glob import glob
from pathlib import Path
import time
from datetime import timedelta
import h5py
import obspy
import obspy.signal.filter
from scipy.signal import decimate
import pandas as pd
import numpy as np

# decimate_data = False


def _proc_filename(fn, year: int, exp_abbr: str):
    if exp_abbr == "C":
        return int(fn.split(exp_abbr)[-1].split(f".{00}")[0])
    if exp_abbr == "S":
        return int(fn.split("." + exp_abbr)[-1].split("..")[0])
    return int(fn.split(exp_abbr)[-1].split(f".{year}")[0])


def import_miniseed(pathname: str, year: int, exp_abbr: str):
    """
    :exp_abbr: the first letter of the experiment
    Assumed the pathname to this type of input file will be
    under specific date folder within CANDAS or CASTOR folders in /data
    """
    common_path_name = pathname + "/ZI."
    files = glob(str(common_path_name) + exp_abbr + "*")
    files = sorted(files)

    t0 = time.time()
    data_columns = []
    for fn in files:
        # traces.append(obspy.read(file)[0]) # Use obspy to read in traces from file streams
        if exp_abbr == "C":
            trace_data = obspy.read(fn)[0].data[155000:165000]
        else:
            trace_data = obspy.read(fn)[0].data
        data_col_name = _proc_filename(fn, year, exp_abbr)
        data_columns.append(pd.Series(trace_data, name=data_col_name))
    t1 = time.time()
    print("Reading traces elapsed time: ", str(timedelta(seconds=t1 - t0))[:-4])

    return pd.concat(data_columns, axis=1)


def get_h5_attributes(filename: str):
    with h5py.File(filename, "r") as fp:
        ds = fp["data"]
        start_time = ds.attrs["start_time"]
        dt = ds.attrs["sampling_rate"]
        dx = ds.attrs["spatial_resolution"]
        print("start time =", start_time)
        print("sampling rate =", dt)
        print("spatial resolution =", dx)
        # for att in list(ds.attrs):
        #    print(att)
    return


def import_h5(filename: str):
    """
    Assumed the full pathname will always be CANDAS2 folder in /data for this type of input file
    """
    # full_pathname = pathname / filename
    with h5py.File(filename, "r") as fp:
        ds = fp["data"]
        data = ds[...]
        df = pd.DataFrame(data).T
    return df


def set_data_limits(data, first_ch=None, last_ch=None, first_time=None, last_time=None):
    if first_ch is None:
        first_ch = data.columns[0]
    if last_ch is None:
        last_ch = data.columns[-1]
    if first_time is None:
        first_time = data.index[0]
    if last_time is None:
        last_time = data.index[-1]

    data = data.loc[first_time:last_time, first_ch:last_ch]
    return data


def filter_imported_data(data: pd.DataFrame, pass_low=3, pass_hi=18, freq=50, decimate_data=False):
    """
    Two step filter:
    1. bandpass using obspy, bandpass limits and frequency of input data needed if different from defaults
    2. common mode subtraction using median
    """
    if decimate_data == True:
        df_fil = data.apply(
            lambda x: (
                obspy.signal.filter.bandpass(
                    np.array(x), freqmin=pass_low, freqmax=pass_hi, df=freq
                )
            )
        )
        if freq != 50:
            df_fil = df_fil.apply(
                lambda x: (decimate(x, int(freq / 50)))
            )  # changing sample frequency for all to 50 Hz
    else:
        df_fil = data.apply(
            lambda x: (
                obspy.signal.filter.bandpass(
                    np.array(x), freqmin=pass_low, freqmax=pass_hi, df=freq
                )
            )
        )

    # median filter
    series_m = df_fil.median(axis=1)
    cm_noise_constants = df_fil.apply(lambda x: np.dot(x, series_m) / np.dot(series_m, series_m))
    df_fil_cm = pd.DataFrame(index=df_fil.index.copy(), columns=df_fil.columns.copy())
    df_fil_cm.columns = df_fil.columns
    for ch, _ in df_fil.items():
        df_fil_cm[ch] = df_fil[ch] - cm_noise_constants[ch] * series_m
    return df_fil_cm


def load_bad_channels(filename: str = None, first_ch: int = None, candas1=False):
    if candas1 is True:
        bad_channels_candas1_2nd_picked_ch_4426 = np.loadtxt(
            "channel_selections/quality_picks_candas1_picked_0724_1805_ch_4426-5988.csv",
            delimiter=",",
            skiprows=1,
        )
        bad_channels_candas1_2nd_picked_ch_4426 = bad_channels_candas1_2nd_picked_ch_4426.astype(
            int
        )
        bad_channels_candas1_2nd_picked_ch_4426.T[0] = (
            bad_channels_candas1_2nd_picked_ch_4426.T[0] + 4426
        )
        bad_channels_candas1_2nd_picked_ch_2701 = np.loadtxt(
            "channel_selections/quality_picks_candas1_0815_picked _jul24_ch_2701-4425.csv",
            delimiter=",",
            skiprows=1,
        )
        bad_channels_candas1_2nd_picked_ch_2701 = bad_channels_candas1_2nd_picked_ch_2701.astype(
            int
        )
        bad_channels_candas1_2nd_picked_ch_2701.T[0] = (
            bad_channels_candas1_2nd_picked_ch_2701.T[0] + 2701
        )
        bad_channels_candas1_2nd_picked_ch_2701 = bad_channels_candas1_2nd_picked_ch_2701[:-1563]
        bad_channels_candas1_2nd_picked_ch_415 = np.loadtxt(
            "channel_selections/quality_picks_candas1_0815_picked_jul_23_ch_415-2700.csv",
            delimiter=",",
            skiprows=1,
        )
        bad_channels_candas1_2nd_picked_ch_415 = bad_channels_candas1_2nd_picked_ch_415.astype(int)
        bad_channels_candas1_2nd_picked_ch_415.T[0] = (
            bad_channels_candas1_2nd_picked_ch_415.T[0] + 415
        )
        bad_channels_candas1_2nd_picked = np.concatenate(
            (
                bad_channels_candas1_2nd_picked_ch_415,
                bad_channels_candas1_2nd_picked_ch_2701,
                bad_channels_candas1_2nd_picked_ch_4426,
            ),
            axis=0,
        )
        return bad_channels_candas1_2nd_picked[
            np.where(bad_channels_candas1_2nd_picked.T[1] == 1)
        ].T[0]
    bad_channels = np.loadtxt(filename, delimiter=",", skiprows=1)
    bad_channels = bad_channels.astype(int)
    bad_channels.T[0] = bad_channels.T[0] + first_ch
    bad_channels = bad_channels[np.where(bad_channels.T[1] == 1)].T[0]
    return bad_channels
