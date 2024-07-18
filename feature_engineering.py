import numpy as np
import pandas as pd
import os
import glob

import time
from datetime import datetime, timedelta

from scipy import signal
from scipy.signal import hilbert, chirp, decimate
from scipy.stats import median_abs_deviation
from matplotlib.mlab import psd
from dtaidistance import dtw
from dtaidistance import dtw_visualisation as dtwvis


# Functions to define beta from Munoz, Soto 2022. 
def rms_w(rs: pd.Series, window=200):
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


def rms_larger_w(rs, window=200, opening = 200):
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


def save_matrix(matrix_string, matrix_data, event_date='', path='results'):
    '''
    Uses np.savetxt to save matrices calculated from PCC in csv format to disk
    '''
    if not os.path.exists(path):
        os.mkdir(path)
    timestamp = datetime.now().strftime("%m%d_%H%M")
    np.savetxt(path + '/' + matrix_string + '_' + event_date + '_' + timestamp + '.csv', matrix_data, delimiter=',')


def create_feature_from_matrix(matrix_data):
    complete_matrix = matrix_data.T + matrix_data
    
    n_ch = complete_matrix.shape[0]
    feature = np.zeros(n_ch)
    
    #Apply RMS to every row of every channel
    for ch in range(n_ch):
        temp = complete_matrix[ch]
        temp = temp[temp != 0]
        feature[ch] = np.sqrt(np.mean(temp**2))

    return feature


def load_matrix(matrix_string, matrix_data, event_date='', path='results'):
    list_of_files = glob.glob(path + '/' + matrix_string + '_' + event_date + '*.csv') # might be more than one of same event, so take latest
    latest_file = max(list_of_files, key=os.path.getctime)
    matrix = np.loadtxt(latest_file, delimiter=',') #load latest, so glob them and order by date


def save_feature(feature_string, feature_data, event_date='', path='results'):
    '''
    Uses np.savetxt to save matrices calculated from PCC in csv format to disk
    '''
    if not os.path.exists(path):
        os.mkdir(path)
    timestamp = datetime.now().strftime("%m%d_%H%M")
    np.savetxt(path + '/' + feature_string + '_' + event_date + '_' + timestamp + '.csv', feature_data, delimiter=',')


def run_pcc(df_in, event_date=''):
    '''
    Input pd.DataFrame channels for PCC, output 6 1D np.arrays that can be used as CQI features.
    Takes anywhere from a few minutes to a few hours to run, depending on channel length and number of channels.
    '''
    n_ch = df_in.shape[1] 
    pcc_matrix, pcc_lags_matrix, pcc_mean_matrix, pcc_median_matrix, pcc_mad_matrix, modified_pcc_matrix = init_matrices(n_ch)

    df_fft = prep_pcc(df_in)
    conj_df_fft = np.conj(df_fft)
    np_fft = df_fft.to_numpy().T
    conj_np_fft = conj_df_fft.to_numpy().T

    t0 = time.time()
    i = 0
    for nprow in np_fft[:-1]:
        i+=1
        rs_matr = nprow.T * conj_np_fft[i:,:]
        corr_matr = np.real(np.fft.ifft(rs_matr.T, 2*rs_matr.shape[1]-1, axis=0)).T
        mean_array = np.mean(corr_matr, axis=1)
        median_array = np.median(corr_matr, axis=1)
        mad_array = median_abs_deviation(corr_matr, axis=1)
        lags_array = np.argmax(np.abs(corr_matr), axis=1) - int(corr_matr.shape[1]/2)
        peak_array = np.max(np.abs(corr_matr), axis=1)
        rms_window_array = np.apply_along_axis(rms_w, 1, corr_matr)
        rms_larger_window_array = np.apply_along_axis(rms_larger_w, 1, corr_matr)

        row_len = len(mean_array)
        pcc_mean_matrix[i-1, n_ch-row_len:] = mean_array
        pcc_median_matrix[i-1, n_ch-row_len:] = median_array
        pcc_mad_matrix[i-1, n_ch-row_len:] = mad_array
        pcc_lags_matrix[i-1, n_ch-row_len:] = lags_array
        pcc_matrix[i-1, n_ch-row_len:] = peak_array/rms_window_array
        modified_pcc_matrix[i-1, n_ch-row_len:] = peak_array/rms_larger_window_array

        if i%100 == 0:
            print("Channel", i, "done")    
            t1 = time.time()
            print('Elapsed time: ', str(timedelta(seconds= t1 - t0))[:-4])

    t1 = time.time()
    print('Elapsed time for PCC: ', str(timedelta(seconds= t1 - t0))[:-4])

    matrix_dict = {'pcc_matrix':pcc_matrix, 'pcc_lags_matrix':pcc_lags_matrix, 'pcc_mean_matrix':pcc_mean_matrix, 
                   'pcc_median_matrix':pcc_median_matrix, 'pcc_mad_matrix':pcc_mad_matrix, 'modified_pcc_matrix':modified_pcc_matrix}
    for key,value in matrix_dict.items():
        save_matrix(key, value, event_date, path='results') #this took hours, gotta make sure it saves properly
        globals()[key.replace("matrix",'feature')] = create_feature_from_matrix(value) #just want to name every variable properly without repeating lines of code
        #save_feature(key, value, event_date, path='results') #ask Aleix for clever usage

        return pcc_feature, pcc_lags_feature, pcc_mean_feature, pcc_median_feature, pcc_mad_feature, modified_pcc_feature


def rms(values):
    return np.sqrt(np.mean(np.square(values)))

def root_amplitude(values):
    return np.square(np.mean(np.sqrt(np.abs(values))))

def mad(values):
    return np.median(np.abs(values - np.median(values)))

def calculate_psd(df_in):
    df_psd = np.zeros([df_in.shape[1], 129])
    for channel, _ in df_in.items():
        power, psd_frequencies = psd(df_in[channel], Fs=50)
        channel = channel - df_in.columns[0]
        df_psd[channel] = power
    df_psd = pd.DataFrame(df_psd.T)
    return df_psd

def freq_avg(channel):
    FFT_data        = np.fft.fft(channel, axis=0)  # Complex values
    N               = len(channel)
    real_part_avg   = 2/N*np.mean(np.real(FFT_data))
    imag_part_avg   = 2/N*np.mean(np.imag(FFT_data))
    vector_averaged = np.abs(real_part_avg+1j*imag_part_avg)
    return vector_averaged


def create_feature_df(df_in, event_date):
    #create dataframe
    df_features = pd.DataFrame()

    #standard features
    df_features['mean'] = df_in.mean()
    df_features['median'] = df_in.median()
    df_features['variance'] = df_in.var()
    df_features['skew'] = df_in.skew()
    df_features['kurtosis'] = df_in.kurtosis()
    df_features['rms'] = df_in.apply(lambda x: rms(x))
    df_features['peak'] = df_in.abs().max()
    df_features['crest-factor'] = df_features['peak']/df_features['rms']
    df_features['Average-rectified-value'] = df_in.abs().mean()
    df_features['stdev'] = df_in.std()
    df_features['root-amplitude'] = df_in.apply(lambda x: root_amplitude(x))
    df_features['margin-factor'] = df_features['peak']/df_features['root-amplitude']
    df_features['impulse-factor'] = df_features['peak']/df_features['Average-rectified-value']
    df_features['waveform-factor'] = df_features['rms']/df_features['Average-rectified-value']
    df_features['shape-factor'] = df_features['rms']/df_features['Average-rectified-value']
    df_features['clearance-factor'] = df_features['peak']/df_features['root-amplitude']

    #beta-derived features
    matrix_dict = {'pcc_matrix':pcc_matrix, 'pcc_lags_matrix':pcc_lags_matrix, 'pcc_mean_matrix':pcc_mean_matrix, 
                   'pcc_median_matrix':pcc_median_matrix, 'pcc_mad_matrix':pcc_mad_matrix, 'modified_pcc_matrix':modified_pcc_matrix}
    for key,value in matrix_dict.items():
        load_matrix(key, value, event_date, path='results') #this took hours, gotta make sure it saves properly
        globals()[key.replace("matrix",'feature')] = create_feature_from_matrix(value) 
    df_features['beta'] = pcc_feature
    df_features['modified-beta'] = modified_phase_cc_ch
    df_features['mean_pcc'] = phase_mean_cc_ch
    df_features['median_pcc'] = phase_median_cc_ch
    df_features['mad_pcc'] = phase_mad_cc_ch
    df_features['lags'] = phase_lags_ch
    df_features['median-absolute-deviation'] = phase_mad_cc_ch
    df_features['detection-significance'] = (df_features['peak'] - df_features['median'])/df_features['median-absolute-deviation']

    #PSD-derived features
    df_psd = calculate_psd(df_in)
    df_features['psd_mean'] = np.array(df_psd.mean())
    df_features['psd_median'] = np.array(df_psd.median())
    df_features['psd_variance'] = np.array(df_psd.var())
    df_features['psd_skew'] = np.array(df_psd.skew())
    df_features['psd_kurtosis'] = np.array(df_psd.kurtosis())
    df_features['psd_rms'] = np.array(df_psd.apply(lambda x: rms(x)))
    df_features['psd_peak'] = np.array(df_psd.abs().max())
    df_features['psd_crest-factor'] = np.array(df_features['psd_peak']/df_features['psd_rms'])

    #FFT-derived features
    df_features['freq-avg'] = df_in.apply(lambda x: freq_avg(x))

    return df_features