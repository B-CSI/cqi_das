from glob import glob
from pathlib import Path
from common import DATA_DIR
import time
from datetime import timedelta
import h5py
import obspy
import pandas as pd


def _proc_filename(fn, year: int, exp_abbr: str):
    return int(fn.split(exp_abbr)[-1].split(f'.{year}')[0])


def import_miniseed(pathname: Path, year: int, exp_abbr: str):
    """
    :exp_abbr: the first letter of the experiment
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
    print('Reading traces elapsed time: ', str(timedelta(seconds= t1 - t0)))
    
    return pd.concat(data_columns, axis=1)


def import_h5(pathname: Path, filename: str):
    '''
    '''
    full_pathname = pathname / filename
    with h5py.File(full_pathname,"r") as fp:
        ds = fp["data"]
        data = ds[...]
    return data


