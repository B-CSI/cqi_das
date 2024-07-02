import numpy as np
import pandas as pd
import os

import time
from datetime import timedelta

from scipy import signal
from scipy.signal import hilbert, chirp, decimate
from scipy.stats import median_abs_deviation
from matplotlib.mlab import psd
from dtaidistance import dtw
from dtaidistance import dtw_visualisation as dtwvis


# Functions to define beta from Munoz, Soto 2022. 
def rms_w(rs: pd.Series, window: int):
    '''
    Definition of the RMS window from the definition of beta from Munoz, Soto 2022. 
    RMS of all phase cross-correlations a user-specified window distance away from the max PCC, not including the max value.
    :rs: is blah 
    '''
    window_center = np.argmax(np.abs(rs))
    window_left = window_center - window
    if window_left < 0:
        window_left = 0
    window_right = window_center + window + 1
    if window_right > len(rs):
        window_right = len(rs)
    return np.sqrt((np.sum(rs[window_left:window_center]**2) + np.sum(rs[window_center+1:window_right]**2))/(2*window))


def rms_larger_w(rs, window, opening = 200):
    '''
    Definition of a RMS window modified from the definition of beta from Munoz, Soto 2022. 
    RMS of all phase cross-correlations a user-specified window distance *and* opening away from the max PCC, to better quantify PCC noise.
    '''
    window_center = np.argmax(np.abs(rs))
    window_left = window_center - window - opening
    if window_left < 0:
        window_left = 0
    window_right = window_center + window + 1 + opening
    if window_right > len(rs):
        window_right = len(rs)
    return np.sqrt((np.sum(rs[window_left:window_center-opening]**2) + np.sum(rs[window_center+1+opening:window_right]**2))/(2*window))


def init_matrices(n_ch):
    '''
    initialize the matrices for phase cross-correlation with zeroes
    '''
    pcc_matrix = np.zeros([n_ch, n_ch])
    pcc_lags_matrix = np.zeros([n_ch,n_ch])
    pcc_mean_matrix = np.zeros([n_ch,n_ch])
    pcc_median_matrix = np.zeros([n_ch,n_ch])
    pcc_mad_matrix = np.zeros([n_ch,n_ch])
    modified_pcc_matrix = np.zeros([n_ch,n_ch])
    return pcc_matrix, pcc_lags_matrix, pcc_mean_matrix, pcc_median_matrix, pcc_mad_matrix, modified_pcc_matrix

def prep_pcc(df_in):
    '''
    Convert channel input to complex form via hilbert transform and fourier transform for fast(ish) PCC
    '''
    df_hilbert = hilbert(df_in.T)
    df_hilbert = df_hilbert/np.abs(df_hilbert)
    df_fft = np.fft.fft(df_hilbert)
    df_fft = pd.DataFrame(df_fft.T)
    return df_fft

def save_matrix(matrix_string, matrix_data, path='data'):
    '''
    Uses np.savetxt to save matrices calculated from PCC in csv format to disk
    '''
    if not os.path.exists(path):
        os.mkdir(path)
    np.save
    np.savetxt(path + '/' + matrix_string + '.csv', matrix_data, delimiter=',')


#phase cross-correlation attempt
def run_pcc(df_in, n_ch):
    '''
    Input channels for PCC, output 6 matrices that can be used for CQI feature engineering.
    Takes anywhere from a few minutes to a few hours to run, depending on channel length and number of channels.
    '''
    t0 = time.time()
    pcc_matrix, pcc_lags_matrix, pcc_mean_matrix, pcc_median_matrix, pcc_mad_matrix, modified_pcc_matrix = init_matrices(n_ch)

    df_fft = prep_pcc(df_in)
    t1 = time.time()

    print('Elapsed time for PCC prep: ', str(timedelta(seconds= t1 - t0)))
    
    t0 = time.time()
    for col1 in df_fft.columns: 
        for col2 in df_fft.columns:
            if col2 > col1:
                signal_a = df_fft[col1]
                signal_b = np.conj(df_fft[col2])
                rs = signal_a * signal_b 
                corrs = np.real(np.fft.ifft(rs, 2*rs.shape[0]-1, axis=0))
                lags = np.argmax(np.abs(corrs)) - int(len(corrs)/2)
                rms_window = rms_w(corrs, 200) 
                rms_larger_window = rms_larger_w(corrs, 200, 200) 
                corr_col1 = col1 - df_fft.columns[0]
                corr_col2 = col2 - df_fft.columns[0]
                pcc_matrix[corr_col1][corr_col2] = lags
                pcc_lags_matrix[corr_col1][corr_col2] = np.mean(corrs)
                pcc_mean_matrix[corr_col1][corr_col2] = np.median(corrs)
                pcc_median_matrix[corr_col1][corr_col2] = median_abs_deviation(corrs)
                pcc_mad_matrix[corr_col1][corr_col2] = np.max(np.abs(corrs))/rms_window
                modified_pcc_matrix[corr_col1][corr_col2] = np.max(np.abs(corrs))/rms_larger_window
    t1 = time.time()
    print('Elapsed time for PCC: ', str(timedelta(seconds= t1 - t0)))

    matrix_dict = {'pcc_matrix': pcc_matrix, 'pcc_lags_matrix':pcc_lags_matrix, 'pcc_mean_matrix':pcc_mean_matrix, 
                   'pcc_median_matrix':pcc_median_matrix, 'pcc_mad_matrix':pcc_mad_matrix, 'modified_pcc_matrix':modified_pcc_matrix}
    for key in matrix_dict:
        save_matrix(key, matrix_dict[key])

    return pcc_matrix, pcc_lags_matrix, pcc_mean_matrix, pcc_median_matrix, pcc_mad_matrix, modified_pcc_matrix
