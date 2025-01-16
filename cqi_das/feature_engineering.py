"""
Copyright (c) 2025, Spanish National Research Council (CSIC)
Developed by the Barcelona Center for Subsurface Imaging (B-CSI)

This source code is subject to the terms of the
GNU Lesser General Public License.


Note: this file is kept for legacy purposes, but contains unclean code.
"""

import numpy as np
import pandas as pd
import os
import glob

import time
from datetime import datetime, timedelta

import librosa
from scipy import signal
from scipy.fft import fft
from scipy.signal import hilbert, find_peaks, peak_prominences
from scipy.stats import median_abs_deviation, kurtosis, entropy
from matplotlib.mlab import psd


# Functions to define beta from Munoz, Soto 2022.
def rms_w(rs: pd.Series, window=200):
    """
    Definition of the RMS window from the definition of beta from Munoz, Soto 2022.
    RMS of all phase cross-correlations a user-specified window distance away from the max PCC, not including the max value.
    :rs: is blah
    """
    window_center = np.argmax(np.abs(rs))
    window_left = window_center - window
    if window_left < 0:
        window_left = 0
    window_right = window_center + window + 1
    if window_right > len(rs):
        window_right = len(rs)
    return np.sqrt(
        (
            np.sum(rs[window_left:window_center] ** 2)
            + np.sum(rs[window_center + 1 : window_right] ** 2)
        )
        / (2 * window)
    )


def rms_larger_w(rs, window=200, opening=200):
    """
    Definition of a RMS window modified from the definition of beta from Munoz, Soto 2022.
    RMS of all phase cross-correlations a user-specified window distance *and* opening away from the max PCC, to better quantify PCC noise.
    """
    window_center = np.argmax(np.abs(rs))
    window_left = window_center - window - opening
    if window_left < 0:
        window_left = 0
    window_right = window_center + window + 1 + opening
    if window_right > len(rs):
        window_right = len(rs)
    return np.sqrt(
        (
            np.sum(rs[window_left : window_center - opening] ** 2)
            + np.sum(rs[window_center + 1 + opening : window_right] ** 2)
        )
        / (2 * window)
    )


def init_matrices(n_ch):
    """
    initialize the matrices for phase cross-correlation with zeroes
    """
    pcc_matrix = np.zeros([n_ch, n_ch])
    pcc_lags_matrix = np.zeros([n_ch, n_ch])
    pcc_mean_matrix = np.zeros([n_ch, n_ch])
    pcc_median_matrix = np.zeros([n_ch, n_ch])
    pcc_mad_matrix = np.zeros([n_ch, n_ch])
    modified_pcc_matrix = np.zeros([n_ch, n_ch])
    return (
        pcc_matrix,
        pcc_lags_matrix,
        pcc_mean_matrix,
        pcc_median_matrix,
        pcc_mad_matrix,
        modified_pcc_matrix,
    )


def prep_pcc(df_in):
    """
    Convert channel input to complex form via hilbert transform and fourier transform for fast(ish) PCC
    """

    df_hilbert = hilbert(df_in.T)
    df_hilbert = df_hilbert / np.abs(df_hilbert)
    df_fft = np.fft.fft(df_hilbert)
    df_fft = pd.DataFrame(df_fft.T)
    return df_fft


def save_matrix(matrix_string, matrix_data, event_date="", path="results"):
    """
    Uses np.savetxt to save matrices calculated from PCC in csv format to disk
    """
    if not os.path.exists(path):
        os.mkdir(path)
    timestamp = datetime.now().strftime("%m%d_%H%M")
    np.savetxt(
        path + "/" + matrix_string + "_" + event_date + "_" + timestamp + ".csv",
        matrix_data,
        delimiter=",",
    )


def create_feature_from_matrix(matrix_data):
    complete_matrix = matrix_data.T + matrix_data

    n_ch = complete_matrix.shape[0]
    feature = np.zeros(n_ch)

    # Apply RMS to every row of every channel
    for ch in range(n_ch):
        temp = complete_matrix[ch]
        temp = temp[temp != 0]
        feature[ch] = np.sqrt(np.mean(temp**2))

    return feature


def load_matrix(matrix_string, event_date="", path="results"):
    list_of_files = glob.glob(
        path + "/" + matrix_string + "*" + event_date + "*.csv"
    )  # might be more than one of same event, so take latest
    latest_file = max(list_of_files, key=os.path.getctime)
    matrix = np.loadtxt(latest_file, delimiter=",")  # load latest, so glob them and order by date
    return matrix


def load_feature(feature_string, event_date="", path="results"):
    list_of_files = glob.glob(
        path + "/" + feature_string + "*" + event_date + "*.csv"
    )  # might be more than one of same event, so take latest
    latest_file = max(list_of_files, key=os.path.getctime)
    feature = np.loadtxt(latest_file, delimiter=",")  # load latest, so glob them and order by date
    return feature


def save_feature(feature_string, feature_data, event_date="", path="results"):
    """
    Uses np.savetxt to save matrices calculated from PCC in csv format to disk
    """
    if not os.path.exists(path):
        os.mkdir(path)
    timestamp = datetime.now().strftime("%m%d_%H%M")
    np.savetxt(
        path + "/" + feature_string + "_" + event_date + "_" + timestamp + ".csv",
        feature_data,
        delimiter=",",
    )


def run_pcc(df_in, event_date=""):
    """
    Input pd.DataFrame channels for PCC, output 6 1D np.arrays that can be used as CQI features.
    Takes anywhere from a few minutes to a few hours to run, depending on channel length and number of channels.
    """
    n_ch = df_in.shape[1]
    (
        pcc_matrix,
        pcc_lags_matrix,
        pcc_mean_matrix,
        pcc_median_matrix,
        pcc_mad_matrix,
        modified_pcc_matrix,
    ) = init_matrices(n_ch)

    df_fft = prep_pcc(df_in)
    conj_df_fft = np.conj(df_fft)
    np_fft = df_fft.to_numpy().T
    conj_np_fft = conj_df_fft.to_numpy().T

    t0 = time.time()
    i = 0
    for nprow in np_fft[:-1]:
        i += 1
        rs_matr = nprow.T * conj_np_fft[i:, :]
        corr_matr = np.real(np.fft.ifft(rs_matr.T, 2 * rs_matr.shape[1] - 1, axis=0)).T
        mean_array = np.mean(corr_matr, axis=1)
        median_array = np.median(corr_matr, axis=1)
        mad_array = median_abs_deviation(corr_matr, axis=1)
        lags_array = np.argmax(np.abs(corr_matr), axis=1) - int(corr_matr.shape[1] / 2)
        peak_array = np.max(np.abs(corr_matr), axis=1)
        rms_window_array = np.apply_along_axis(rms_w, 1, corr_matr)
        rms_larger_window_array = np.apply_along_axis(rms_larger_w, 1, corr_matr)

        row_len = len(mean_array)
        pcc_mean_matrix[i - 1, n_ch - row_len :] = mean_array
        pcc_median_matrix[i - 1, n_ch - row_len :] = median_array
        pcc_mad_matrix[i - 1, n_ch - row_len :] = mad_array
        pcc_lags_matrix[i - 1, n_ch - row_len :] = lags_array
        pcc_matrix[i - 1, n_ch - row_len :] = peak_array / rms_window_array
        modified_pcc_matrix[i - 1, n_ch - row_len :] = peak_array / rms_larger_window_array

        if i % 100 == 0:
            print("Channel", i, "done")
            t1 = time.time()
            print("Elapsed time: ", str(timedelta(seconds=t1 - t0))[:-4])

    t1 = time.time()
    print("Elapsed time for PCC: ", str(timedelta(seconds=t1 - t0))[:-4])

    matrix_dict = {
        "pcc_matrix": pcc_matrix,
        "pcc_lags_matrix": pcc_lags_matrix,
        "pcc_mean_matrix": pcc_mean_matrix,
        "pcc_median_matrix": pcc_median_matrix,
        "pcc_mad_matrix": pcc_mad_matrix,
        "modified_pcc_matrix": modified_pcc_matrix,
    }
    features = {}
    for key, value in matrix_dict.items():
        # save_matrix(key, value, event_date, path='results') #this took hours, gotta make sure it saves properly
        # globals()[key.replace("matrix",'feature')] = create_feature_from_matrix(value) #just want to name every variable properly without repeating lines of code
        new_key = key.replace("matrix", "feature")
        features[new_key] = create_feature_from_matrix(value)
        save_feature(new_key, features[new_key], event_date, path="results")

    return features


def rms(values):
    return np.sqrt(np.mean(np.square(values)))


def root_amplitude(values):
    return np.square(np.mean(np.sqrt(np.abs(values))))


def mad(values):
    return np.median(np.abs(values - np.median(values)))


def time_of_event_max(data):
    # adapted from ChatGPT
    channel_maxima = data.apply(lambda x: np.argmax(x))

    # Create histogram
    counts, bin_edges = np.histogram(channel_maxima, bins=500)

    # Find the bin with the maximum count
    max_count_index = np.argmax(counts)

    # Calculate the x-axis value (midpoint of the bin) corresponding to the maximum count
    return int((bin_edges[max_count_index] + bin_edges[max_count_index + 1]) / 2)


def calculate_mfccs(values):
    mfccs = librosa.feature.mfcc(y=values.to_numpy(), sr=50, n_mfcc=13)
    return np.mean(mfccs, axis=1), np.max(mfccs, axis=1)


def calculate_psd(df_in):
    df_psd = np.zeros([df_in.shape[1], 129])
    for channel, _ in df_in.items():
        power, psd_frequencies = psd(df_in[channel], Fs=50)
        channel = channel - df_in.columns[0]
        df_psd[channel] = power
    df_psd = pd.DataFrame(df_psd.T)
    return df_psd


def freq_avg(channel):
    FFT_data = np.fft.fft(channel, axis=0)  # Complex values
    N = len(channel)
    real_part_avg = 2 / N * np.mean(np.real(FFT_data))
    imag_part_avg = 2 / N * np.mean(np.imag(FFT_data))
    vector_averaged = np.abs(real_part_avg + 1j * imag_part_avg)
    return vector_averaged


def wave_mode(df_in, window_center, window_length=200):
    c = window_center
    w_start = c - window_length // 2
    w_end = c + window_length // 2 + 1
    if w_start < df_in.columns[0]:
        w_start = df_in.columns[0]
        w_end = df_in.columns[window_length]
    if w_end > df_in.columns[-1]:
        w_start = df_in.columns[-window_length]
        w_end = df_in.columns[-1]

    return time_of_event_max(df_in.loc[:, w_start:w_end])


def shortest_distance(points1, points2):  # chatgpt
    shortest_dist = np.inf
    for p1 in points1:
        for p2 in points2:
            dist = np.linalg.norm(p1 - p2)  # Euclidean distance
            if dist < shortest_dist:
                shortest_dist = dist

    return shortest_dist


def get_slope(seq):
    temp_seq = np.sort(seq)
    return (temp_seq[-1] - temp_seq[0]) / len(temp_seq)


def get_alt_slope(seq):
    seq = np.array(seq)
    temp_seq = np.sort(seq)
    if np.argmax(seq) < np.argmin(seq):
        return (temp_seq[0] - temp_seq[-1]) / len(temp_seq)
    else:
        return (temp_seq[-1] - temp_seq[0]) / len(temp_seq)


# there should be a 50% overlap on the windows! So the rms value can be assigned to the point in the middle
def make_envelope(data, window_size):
    test_rms_list = []
    for seq in np.lib.stride_tricks.sliding_window_view(data, window_size)[
        :: window_size // 2, :
    ]:  # 50% overlap
        test_rms_list.append(rms(seq))
    return test_rms_list


def get_prominence_factor(envelope_channel, num_peaks):
    for i in reversed(range(101)):
        peaks, properties = find_peaks(envelope_channel, prominence=np.max(envelope_channel) / i)
        if len(properties["prominences"]) <= num_peaks and len(properties["prominences"]) > 0:
            return i
        if i == 1:
            return 0


def create_feature_df(df_in):
    # create dataframe
    df_features = pd.DataFrame()

    # standard features
    df_features["mean"] = df_in.mean()
    df_features["median"] = df_in.median()
    df_features["variance"] = df_in.var()
    df_features["skew"] = df_in.skew()
    df_features["kurtosis"] = df_in.kurtosis()
    df_features["rms"] = df_in.apply(lambda x: rms(x))
    df_features["peak"] = df_in.abs().max()
    df_features["crest-factor"] = df_features["peak"] / df_features["rms"]
    df_features["Average-rectified-value"] = df_in.abs().mean()
    df_features["stdev"] = df_in.std()
    df_features["root-amplitude"] = df_in.apply(lambda x: root_amplitude(x))
    df_features["margin-factor"] = df_features["peak"] / df_features["root-amplitude"]
    df_features["impulse-factor"] = df_features["peak"] / df_features["Average-rectified-value"]
    df_features["waveform-factor"] = df_features["rms"] / df_features["mean"]
    df_features["shape-factor"] = df_features["rms"] / df_features["Average-rectified-value"]
    df_features["clearance-factor"] = df_features["peak"] / df_features["root-amplitude"]
    df_features["median-absolute-deviation"] = df_in.apply(lambda x: mad(x))
    df_features["detection-significance"] = (
        df_features["peak"] - df_features["median"]
    ) / df_features["median-absolute-deviation"]

    # if dataframe is normalized, then drop(columns=['mean', 'variance', 'rms', 'stdev', 'freq-rms', "waveform-factor"])

    # new features
    t0 = time.time()
    kurt_window = 250
    grad_window = 25
    stack_window = 100
    kurtosis_df = df_in.apply(
        lambda x: kurtosis(
            np.lib.stride_tricks.sliding_window_view(x, window_shape=kurt_window), axis=1
        )
    )
    gradient_kurtosis_df = kurtosis_df.apply(lambda x: np.gradient(x))
    t1 = time.time()
    print("did simple kurtosis: ", str(timedelta(seconds=t1 - t0))[:-4])
    gentle_gradient_kurtosis_df = kurtosis_df.apply(
        lambda x: np.apply_along_axis(
            get_alt_slope, axis=1, arr=np.lib.stride_tricks.sliding_window_view(x, grad_window)
        ),
        axis=0,
    )
    t1 = time.time()
    print("applied gentle gradient: ", str(timedelta(seconds=t1 - t0))[:-4])
    stacked_channels_df = df_in.T.rolling(window=stack_window).sum().T
    t1 = time.time()
    print("stacked channels: ", str(timedelta(seconds=t1 - t0))[:-4])
    stacked_kurtosis_df = stacked_channels_df.apply(
        lambda x: kurtosis(
            np.lib.stride_tricks.sliding_window_view(x, window_shape=kurt_window), axis=1
        )
    )
    t1 = time.time()
    print("did kurtosis on stacked channels: ", str(timedelta(seconds=t1 - t0))[:-4])
    stacked_gradient_kurtosis_df = stacked_kurtosis_df.apply(lambda x: np.gradient(x))
    stacked_kurtosis_gradients = []
    for seq in np.lib.stride_tricks.sliding_window_view(
        stacked_gradient_kurtosis_df.columns, window_shape=stack_window
    )[
        ::1
    ]:  # was grad_window?
        stacked_kurtosis_gradients.append(
            np.median(np.argmax(stacked_gradient_kurtosis_df.loc[:, seq], axis=0))
        )
    stacked_kurtosis_gradients = np.array(stacked_kurtosis_gradients) + kurt_window
    extension_length = df_in.shape[1] - len(stacked_kurtosis_gradients)
    stacked_kurtosis_gradients = np.concatenate(
        [stacked_kurtosis_gradients, np.full(extension_length, stacked_kurtosis_gradients[-1])]
    )

    gentle_stacked_gradient_kurtosis_df = stacked_kurtosis_df.apply(
        lambda x: np.apply_along_axis(
            get_alt_slope, axis=1, arr=np.lib.stride_tricks.sliding_window_view(x, grad_window)
        ),
        axis=0,
    )
    gentle_stacked_kurtosis_gradients = []
    for seq in np.lib.stride_tricks.sliding_window_view(
        gentle_stacked_gradient_kurtosis_df.columns, window_shape=stack_window
    )[::1]:
        gentle_stacked_kurtosis_gradients.append(
            np.median(np.argmax(gentle_stacked_gradient_kurtosis_df.loc[:, seq], axis=0))
        )
    gentle_stacked_kurtosis_gradients = np.array(gentle_stacked_kurtosis_gradients) + kurt_window
    extension_length = df_in.shape[1] - len(gentle_stacked_kurtosis_gradients)
    gentle_stacked_kurtosis_gradients = np.concatenate(
        [
            gentle_stacked_kurtosis_gradients,
            np.full(extension_length, gentle_stacked_kurtosis_gradients[-1]),
        ]
    )
    t1 = time.time()
    print("finished gentle kurtosis: ", str(timedelta(seconds=t1 - t0))[:-4])

    envelope_df = df_in.apply(lambda x: make_envelope(x, 100))  # 2x sample frequency
    t1 = time.time()
    print("created envelope: ", str(timedelta(seconds=t1 - t0))[:-4])
    gauge_length = 10
    df_features["dist-from-interrogator"] = df_in.columns * gauge_length

    # take min of diff of top three
    df_features["dist-from-signal-max"] = np.min(
        (
            (
                np.abs(
                    gradient_kurtosis_df.apply(lambda x: np.argsort(np.array(x))[-1])
                    - stacked_kurtosis_gradients
                )
            ),
            (
                np.abs(
                    gradient_kurtosis_df.apply(lambda x: np.argsort(np.array(x))[-2])
                    - stacked_kurtosis_gradients
                )
            ),
            (
                np.abs(
                    gradient_kurtosis_df.apply(lambda x: np.argsort(np.array(x))[-3])
                    - stacked_kurtosis_gradients
                )
            ),
        ),
        axis=0,
    )
    df_features["gentle-dist-from-signal-max"] = np.min(
        (
            np.abs(
                gentle_gradient_kurtosis_df.apply(lambda x: np.argsort(np.array(x))[-1])
                - gentle_stacked_kurtosis_gradients
            ),
            np.abs(
                gentle_gradient_kurtosis_df.apply(lambda x: np.argsort(np.array(x))[-2])
                - gentle_stacked_kurtosis_gradients
            ),
            np.abs(
                gentle_gradient_kurtosis_df.apply(lambda x: np.argsort(np.array(x))[-3])
                - gentle_stacked_kurtosis_gradients
            ),
        ),
        axis=0,
    )

    df_features["3-peak-prominence-factor"] = envelope_df.apply(
        lambda x: get_prominence_factor(x, 3)
    )
    df_features["1-peak-prominence-factor"] = envelope_df.apply(
        lambda x: get_prominence_factor(x, 1)
    )

    df_features["env-mean"] = envelope_df.mean()
    df_features["env-median"] = envelope_df.median()
    df_features["env-variance"] = envelope_df.var()
    df_features["env-skew"] = envelope_df.skew()
    df_features["env-kurtosis"] = envelope_df.kurtosis()
    df_features["env-entropy"] = envelope_df.apply(lambda x: entropy(x))
    df_features["env-rms"] = envelope_df.apply(lambda x: rms(x))
    df_features["env-peak"] = envelope_df.abs().max()
    df_features["env-crest-factor"] = df_features["env-peak"] / df_features["env-rms"]
    df_features["env-Average-rectified-value"] = envelope_df.abs().mean()
    df_features["env-stdev"] = envelope_df.std()
    df_features["env-root-amplitude"] = envelope_df.apply(lambda x: root_amplitude(x))
    df_features["env-margin-factor"] = df_features["env-peak"] / df_features["env-root-amplitude"]
    df_features["env-impulse-factor"] = (
        df_features["env-peak"] / df_features["env-Average-rectified-value"]
    )
    df_features["env-waveform-factor"] = df_features["env-rms"] / df_features["env-mean"]
    df_features["env-shape-factor"] = (
        df_features["env-rms"] / df_features["env-Average-rectified-value"]
    )
    df_features["env-clearance-factor"] = (
        df_features["env-peak"] / df_features["env-root-amplitude"]
    )
    df_features["env-median-absolute-deviation"] = envelope_df.apply(lambda x: mad(x))
    df_features["env-detection-significance"] = (
        df_features["env-peak"] - df_features["env-median"]
    ) / df_features["env-median-absolute-deviation"]

    # what about kurtosis *of* rms envelope? (maybe ask hugo how his works?) - d/dt kurtosis, windowed
    # one single value across recorded experiments for how contrasting the experiment is (how shitty the channels are overall)
    # -- maybe for each channel the max/min ratio? Or basically crest factor but overall for only selected good channels
    # are both neighbors (or whole region) good/bad %;
    # proportion of 10 around (+/-5) that are good/bad
    # CHECK #delta kurtosis of freq x delta kurtosis of amplitude?
    t1 = time.time()
    print("created normal features ", str(timedelta(seconds=t1 - t0))[:-4])

    # mfcc features
    mfccs_mean = df_in.apply(lambda x: calculate_mfccs(x)[0])
    mfccs_max = df_in.apply(lambda x: calculate_mfccs(x)[1])
    print("created mfccs")
    df_features["mfcc0_mean"] = mfccs_mean.loc[0, :]
    df_features["mfcc1_mean"] = mfccs_mean.loc[1, :]
    df_features["mfcc2_mean"] = mfccs_mean.loc[2, :]
    df_features["mfcc3_mean"] = mfccs_mean.loc[3, :]
    df_features["mfcc4_mean"] = mfccs_mean.loc[4, :]
    df_features["mfcc5_mean"] = mfccs_mean.loc[5, :]
    df_features["mfcc6_mean"] = mfccs_mean.loc[6, :]
    df_features["mfcc7_mean"] = mfccs_mean.loc[7, :]
    df_features["mfcc8_mean"] = mfccs_mean.loc[8, :]
    df_features["mfcc9_mean"] = mfccs_mean.loc[9, :]
    df_features["mfcc10_mean"] = mfccs_mean.loc[10, :]
    df_features["mfcc11_mean"] = mfccs_mean.loc[11, :]
    df_features["mfcc12_mean"] = mfccs_mean.loc[12, :]
    df_features["mfcc0_max"] = mfccs_max.loc[0, :]
    df_features["mfcc1_max"] = mfccs_max.loc[1, :]
    df_features["mfcc2_max"] = mfccs_max.loc[2, :]
    df_features["mfcc3_max"] = mfccs_max.loc[3, :]
    df_features["mfcc4_max"] = mfccs_max.loc[4, :]
    df_features["mfcc5_max"] = mfccs_max.loc[5, :]
    df_features["mfcc6_max"] = mfccs_max.loc[6, :]
    df_features["mfcc7_max"] = mfccs_max.loc[7, :]
    df_features["mfcc8_max"] = mfccs_max.loc[8, :]
    df_features["mfcc9_max"] = mfccs_max.loc[9, :]
    df_features["mfcc10_max"] = mfccs_max.loc[10, :]
    df_features["mfcc11_max"] = mfccs_max.loc[11, :]
    df_features["mfcc12_max"] = mfccs_max.loc[12, :]
    # and now applied to the envelope:
    env_mfccs_mean = envelope_df.apply(lambda x: calculate_mfccs(x)[0])
    env_mfccs_max = envelope_df.apply(lambda x: calculate_mfccs(x)[1])
    t1 = time.time()
    print("created envelope mfccs ", str(timedelta(seconds=t1 - t0))[:-4])
    df_features["env_mfcc0_mean"] = env_mfccs_mean.loc[0, :]
    df_features["env_mfcc1_mean"] = env_mfccs_mean.loc[1, :]
    df_features["env_mfcc2_mean"] = env_mfccs_mean.loc[2, :]
    df_features["env_mfcc3_mean"] = env_mfccs_mean.loc[3, :]
    df_features["env_mfcc4_mean"] = env_mfccs_mean.loc[4, :]
    df_features["env_mfcc5_mean"] = env_mfccs_mean.loc[5, :]
    df_features["env_mfcc6_mean"] = env_mfccs_mean.loc[6, :]
    df_features["env_mfcc7_mean"] = env_mfccs_mean.loc[7, :]
    df_features["env_mfcc8_mean"] = env_mfccs_mean.loc[8, :]
    df_features["env_mfcc9_mean"] = env_mfccs_mean.loc[9, :]
    df_features["env_mfcc10_mean"] = env_mfccs_mean.loc[10, :]
    df_features["env_mfcc11_mean"] = env_mfccs_mean.loc[11, :]
    df_features["env_mfcc12_mean"] = env_mfccs_mean.loc[12, :]
    df_features["env_mfcc0_max"] = env_mfccs_max.loc[0, :]
    df_features["env_mfcc1_max"] = env_mfccs_max.loc[1, :]
    df_features["env_mfcc2_max"] = env_mfccs_max.loc[2, :]
    df_features["env_mfcc3_max"] = env_mfccs_max.loc[3, :]
    df_features["env_mfcc4_max"] = env_mfccs_max.loc[4, :]
    df_features["env_mfcc5_max"] = env_mfccs_max.loc[5, :]
    df_features["env_mfcc6_max"] = env_mfccs_max.loc[6, :]
    df_features["env_mfcc7_max"] = env_mfccs_max.loc[7, :]
    df_features["env_mfcc8_max"] = env_mfccs_max.loc[8, :]
    df_features["env_mfcc9_max"] = env_mfccs_max.loc[9, :]
    df_features["env_mfcc10_max"] = env_mfccs_max.loc[10, :]
    df_features["env_mfcc11_max"] = env_mfccs_max.loc[11, :]
    df_features["env_mfcc12_max"] = env_mfccs_max.loc[12, :]

    # PSD-derived features
    df_psd = calculate_psd(df_in)
    t1 = time.time()
    print("created psd ", str(timedelta(seconds=t1 - t0))[:-4])
    df_features["psd-mean"] = np.array(df_psd.mean())
    df_features["psd-median"] = np.array(df_psd.median())
    df_features["psd-variance"] = np.array(df_psd.var())
    df_features["psd-skew"] = np.array(df_psd.skew())
    df_features["psd-kurtosis"] = np.array(df_psd.kurtosis())
    df_features["psd-entropy"] = np.array(df_psd.apply(lambda x: entropy(x)))
    df_features["psd-rms"] = np.array(df_psd.apply(lambda x: rms(x)))
    df_features["psd-peak"] = np.array(df_psd.abs().max())
    df_features["psd-crest-factor"] = np.array(df_features["psd-peak"] / df_features["psd-rms"])
    # applied to envelope
    df_env_psd = calculate_psd(envelope_df)
    t1 = time.time()
    print("created psd of envelope ", str(timedelta(seconds=t1 - t0))[:-4])
    df_features["env_psd-mean"] = np.array(df_env_psd.mean())
    df_features["env_psd-median"] = np.array(df_env_psd.median())
    df_features["env_psd-variance"] = np.array(df_env_psd.var())
    df_features["env_psd-skew"] = np.array(df_env_psd.skew())
    df_features["env_psd-kurtosis"] = np.array(df_env_psd.kurtosis())
    df_features["env-psd-entropy"] = np.array(df_env_psd.apply(lambda x: entropy(x)))
    df_features["env_psd-rms"] = np.array(df_env_psd.apply(lambda x: rms(x)))
    df_features["env_psd-peak"] = np.array(df_env_psd.abs().max())
    df_features["env_psd-crest-factor"] = np.array(
        df_features["env_psd-peak"] / df_features["env_psd-rms"]
    )

    # FFT-derived features
    df_fft = df_in.apply(lambda x: fft(x.values)).abs()
    df_features["freq-avg"] = df_in.apply(lambda x: freq_avg(x))
    df_features["freq-median"] = np.array(df_fft.median())
    df_features["freq-variance"] = np.array(df_fft.var())
    df_features["freq-skew"] = np.array(df_fft.skew())
    df_features["freq-kurtosis"] = np.array(df_fft.kurtosis())
    df_features["freq-entropy"] = np.array(df_fft.apply(lambda x: entropy(x)))
    df_features["freq-rms"] = np.array(df_fft.apply(lambda x: rms(x)))
    df_features["freq-peak"] = np.array(df_fft.max())
    df_features["freq-crest-factor"] = np.array(df_features["freq-peak"] / df_features["freq-rms"])
    # df_features['two_kurtoses'] = df_features['freq-kurtosis'] * df_features['kurtosis'] #forgot to make this its own feature

    # applied to envelope
    df_env_fft = envelope_df.apply(lambda x: fft(x.values)).abs()
    df_features["env_freq-avg"] = envelope_df.apply(lambda x: freq_avg(x))
    df_features["env_freq-median"] = np.array(df_env_fft.median())
    df_features["env_freq-variance"] = np.array(df_env_fft.var())
    df_features["env_freq-skew"] = np.array(df_env_fft.skew())
    df_features["env_freq-kurtosis"] = np.array(df_env_fft.kurtosis())
    df_features["env-freq-entropy"] = np.array(df_env_fft.apply(lambda x: entropy(x)))
    df_features["env_freq-rms"] = np.array(df_env_fft.apply(lambda x: rms(x)))
    df_features["env_freq-peak"] = np.array(df_env_fft.max())
    df_features["env_freq-crest-factor"] = np.array(
        df_features["env_freq-peak"] / df_features["env_freq-rms"]
    )
    df_features["two_kurtoses"] = df_features["env_freq-kurtosis"] * df_features["kurtosis"]

    return df_features


def add_beta_features(df_features, path="results", event_date=""):
    # beta-derived features
    # these are deprecated; just load_feature in future
    pcc_feature = create_feature_from_matrix(load_matrix("pcc_matrix", event_date, path))
    pcc_lags_feature = create_feature_from_matrix(load_matrix("pcc_lags_matrix", event_date, path))
    pcc_mean_feature = create_feature_from_matrix(load_matrix("pcc_mad_matrix", event_date, path))
    pcc_median_feature = create_feature_from_matrix(
        load_matrix("pcc_median_matrix", event_date, path)
    )
    pcc_mad_feature = create_feature_from_matrix(load_matrix("pcc_mad_matrix", event_date, path))
    modified_pcc_feature = create_feature_from_matrix(
        load_matrix("modified_pcc_matrix", event_date, path)
    )

    df_features["beta"] = pcc_feature
    df_features["modified-beta"] = modified_pcc_feature
    df_features["mean_pcc"] = pcc_mean_feature
    df_features["median_pcc"] = pcc_median_feature
    df_features["mad_pcc"] = pcc_mad_feature
    df_features["lags"] = pcc_lags_feature

    return df_features
