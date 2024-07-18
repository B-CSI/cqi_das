from glob import glob
from pathlib import Path
from common import DATA_DIR
import time
from datetime import timedelta
import h5py
import obspy
import obspy.signal.filter
from scipy.signal import decimate
import pandas as pd
import numpy as np

#decimate_data = False

def _proc_filename(fn, year: int, exp_abbr: str):
    return int(fn.split(exp_abbr)[-1].split(f'.{year}')[0])


def import_miniseed(pathname: Path, year: int, exp_abbr: str):
    """
    :exp_abbr: the first letter of the experiment
    Assumed the pathname to this type of input file will be under specific date folder within CANDAS or CASTOR folders in /data
    """
    common_path_name = pathname / "ZI."
    files = glob(str(common_path_name) + exp_abbr + '*') 
    files = sorted(files)
    
    t0 = time.time()
    data_columns = []
    for fn in files:
    # traces.append(obspy.read(file)[0]) # Use obspy to read in traces from file streams
        trace_data = obspy.read(fn)[0].data
        data_col_name = _proc_filename(fn, year, exp_abbr)
        data_columns.append(pd.Series(trace_data, name=data_col_name))
    t1 = time.time()
    print('Reading traces elapsed time: ', str(timedelta(seconds= t1 - t0))[:-4])
    
    return pd.concat(data_columns, axis=1)


def import_h5(pathname: Path, filename: str):
    '''
    Assumed the full pathname will always be CANDAS2 folder in /data for this type of input file
    '''
    full_pathname = pathname / filename
    with h5py.File(full_pathname,"r") as fp:
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


def filter_imported_data(data: pd.DataFrame,pass_low=3, pass_hi=18, freq=50, decimate_data=False):
    '''
    Two step filter:
    1. bandpass using obspy, bandpass limits and frequency of input data needed if different from defaults
    2. common mode subtraction using median
    '''
    df_fil = data.apply(lambda x: (obspy.signal.filter.bandpass(np.array(x), freqmin=pass_low, freqmax=pass_hi, df=50))) 
    if decimate_data == True:
        df_fil = data.apply(lambda x: (obspy.signal.filter.bandpass(np.array(x), freqmin=pass_low, freqmax=12.5, df=50))) 
        df_fil = df_fil.apply(lambda x: (decimate(x, 2))) 

    #median filter
    series_m = df_fil.median(axis=1)
    cm_noise_constants = df_fil.apply(lambda x: np.dot(x,series_m)/np.dot(series_m, series_m))
    df_fil_cm = pd.DataFrame(index=df_fil.index.copy(),columns=df_fil.columns.copy())
    df_fil_cm.columns = df_fil.columns
    for ch, _ in df_fil.items():
        df_fil_cm[ch] = df_fil[ch] - cm_noise_constants[ch]*series_m
    return df_fil_cm