"""
Copyright (c) 2025, Spanish National Research Council (CSIC)
Developed by the Barcelona Center for Subsurface Imaging (B-CSI)

This source code is subject to the terms of the
GNU Lesser General Public License.
"""

import numpy as np
import pandas as pd
import librosa
from scipy.fft import fft
from scipy.signal import find_peaks
from scipy.stats import entropy, kurtosis
from matplotlib.mlab import psd


def rms(values: np.ndarray) -> float:
    """
    Compute the root-mean-square (RMS) of an array.

    Parameters
    ----------
    values : np.ndarray
        Input array.

    Returns
    -------
    float
        RMS value of the input array.
    """
    return np.sqrt(np.mean(values**2))


def root_amplitude(values: np.ndarray) -> float:
    """
    Compute the mean of the element-wise square roots, then square it.

    Parameters
    ----------
    values : np.ndarray
        Input array.

    Returns
    -------
    float
        Result of (mean of sqrt(abs(values)))^2.
    """
    return np.square(np.mean(np.sqrt(np.abs(values))))


def make_envelope(data: np.ndarray, window_size: int) -> list:
    """
    Create an 'envelope' by computing the RMS of overlapping windows
    with 50% overlap.

    Parameters
    ----------
    data : np.ndarray
        1D signal array.
    window_size : int
        Size of the sliding window used to compute RMS segments.

    Returns
    -------
    list
        A list of RMS values for each window (50% overlap).
    """
    # Use stride_tricks to create overlapping windows
    # Step of window_size // 2 => 50% overlap
    result = []
    for seq in np.lib.stride_tricks.sliding_window_view(data, window_size)[:: window_size // 2, :]:
        result.append(rms(seq))
    return result


def freq_avg(channel: np.ndarray) -> float:
    """
    Compute a simple 'vector-averaged' spectrum magnitude
    by averaging the real and imaginary parts of the FFT.

    Parameters
    ----------
    channel : np.ndarray
        1D signal array.

    Returns
    -------
    float
        Vector-averaged magnitude (abs of mean real + j*imag).
    """
    FFT_data = np.fft.fft(channel, axis=0)
    N = len(channel)
    real_part_avg = 2 / N * np.mean(np.real(FFT_data))
    imag_part_avg = 2 / N * np.mean(np.imag(FFT_data))
    return np.abs(real_part_avg + 1j * imag_part_avg)


def calculate_psd(df_in: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate the power spectral density (PSD) for each channel in a DataFrame.
    Uses `matplotlib.mlab.psd`.

    Parameters
    ----------
    df_in : pd.DataFrame
        Columns represent channels, index represents time samples.

    Returns
    -------
    pd.DataFrame
        Each column corresponds to one channel's PSD values.
    """
    n_channels = df_in.shape[1]
    # For a 50 Hz sampling rate, psd(...) returns 129 frequency bins by default.
    df_psd = np.zeros([n_channels, 129])

    for idx, channel in enumerate(df_in.columns):
        power, _ = psd(df_in[channel].values, Fs=50)
        df_psd[idx] = power

    return pd.DataFrame(df_psd.T, columns=df_in.columns)


def get_prominence_factor_bs(envelope_channel: np.ndarray, num_peaks: int) -> int:
    """
    Binary search approach to find a 'peak-prominence-factor'.
    We search for a `prominence` value that yields between 1 and `num_peaks` peaks.

    Parameters
    ----------
    envelope_channel : np.ndarray
        The envelope signal for a single channel.
    num_peaks : int
        The number of peaks we aim to find (at most).

    Returns
    -------
    int
        Largest integer factor that meets the number of peaks criterion.
    """
    max_val = np.max(envelope_channel)
    left, right = 1, 100
    best = 0

    while left <= right:
        mid = (left + right) // 2
        # `prominence = max_val / mid` is our trial threshold
        _, properties = find_peaks(envelope_channel, prominence=max_val / mid)
        c = len(properties["prominences"])

        if 1 <= c <= num_peaks:
            best = mid
            left = mid + 1
        elif c == 0:
            left = mid + 1
        else:
            right = mid - 1

    return best


def calculate_selected_features(data: pd.DataFrame) -> pd.DataFrame:
    """
    Compute 17 features from each channel in `data`, in the exact order specified:

    1.  env-impulse-factor       (Impulse Factor of RMS Envelope)
    2.  env-margin-factor        (Margin Factor of RMS Envelope)
    3.  env-peak-time            (Peak time of RMS envelope)
    4.  env_freq-avg             (Average Frequency Amplitude of RMS Envelope)
    5.  psd-kurtosis             (Kurtosis of Power Spectral Density)
    6.  env-variance             (Amplitude Variance of RMS Envelope)
    7.  1-peak-prominence-factor (Prominence Factor of the 1st Highest Peak)
    8.  env-psd-entropy          (Entropy of the PSD of the RMS Envelope)
    9.  psd-entropy              (Entropy of the PSD)
    10. 3-peak-prominence-factor (Prominence Factor of the 3rd Highest Peak)
    11. psd-skew                 (Skewness of the PSD)
    12. env-median               (Median Amplitude of the RMS Envelope)
    13. env-clearance-factor     (Clearance Factor of the RMS Envelope)
    14. env-rms                  (RMS of the Amplitude of the RMS Envelope)
    15. freq-skew                (Skewness of the frequency signal)
    16. mfcc1_mean               (Mean of first significant MFCC Coefficient)
    17. mfcc1_max                (Maximum of first significant MFCC Coefficient)

    Parameters
    ----------
    data : pd.DataFrame
        Each column is one channel. Rows represent time samples.

    Returns
    -------
    pd.DataFrame
        A DataFrame of shape (n_channels, 17) with the above-named columns.
    """
    # Create a DataFrame to store feature results
    df_features = pd.DataFrame()

    # Create an 'envelope' DataFrame: each column is a list of envelope values per channel
    envelope_df = data.apply(lambda x: make_envelope(x, window_size=100))

    # Peak Prominence (using a binary-search approach)
    #    - "3-peak-prominence-factor": tries to find up to 3 peaks
    #    - "1-peak-prominence-factor": tries to find 1 peak
    df_features["3-peak-prominence-factor"] = envelope_df.apply(
        lambda x: get_prominence_factor_bs(np.array(x), num_peaks=3)
    )
    df_features["1-peak-prominence-factor"] = envelope_df.apply(
        lambda x: get_prominence_factor_bs(np.array(x), num_peaks=1)
    )

    # Basic envelope statistics
    df_features["env-median"] = envelope_df.apply(np.median)
    df_features["env-variance"] = envelope_df.apply(np.var)
    df_features["env-rms"] = envelope_df.apply(lambda x: rms(np.array(x)))
    df_features["env-peak"] = envelope_df.apply(lambda x: np.max(np.abs(x)))

    # Average rectified value of the envelope
    df_features["env-Average-rectified-value"] = envelope_df.apply(lambda x: np.mean(np.abs(x)))

    # Root amplitude of envelope
    df_features["env-root-amplitude"] = envelope_df.apply(lambda x: root_amplitude(np.array(x)))

    # Envelope-based dimensionless factors
    df_features["env-margin-factor"] = df_features["env-peak"] / df_features["env-root-amplitude"]
    df_features["env-impulse-factor"] = (
        df_features["env-peak"] / df_features["env-Average-rectified-value"]
    )
    df_features["env-clearance-factor"] = df_features["env-margin-factor"]

    # PSD-based features for original data
    df_psd = calculate_psd(data)
    df_features["psd-entropy"] = df_psd.apply(lambda x: entropy(x))
    df_features["psd-kurtosis"] = df_psd.apply(lambda x: kurtosis(x))
    df_features["psd-skew"] = df_psd.apply(lambda x: x.skew())

    # PSD-based feature for the envelope
    df_env_psd = calculate_psd(envelope_df)
    df_features["env-psd-entropy"] = df_env_psd.apply(lambda x: entropy(x))

    # Simple FFT-based skew for the original data
    df_fft = data.apply(lambda col: np.abs(fft(col.values)))
    df_features["freq-skew"] = df_fft.apply(lambda x: pd.Series(x).skew())

    # Vector-average of envelope in frequency domain
    df_features["env_freq-avg"] = envelope_df.apply(lambda x: freq_avg(np.array(x)))

    # MFCC Features
    mfcc_means = []
    mfcc_maxes = []
    for channel_name in data.columns:
        y = data[channel_name].values
        if len(y) < 2048:  # pad for FFT
            y_pad = np.pad(y, (0, 2048 - len(y)))
        else:
            y_pad = y
        # Compute 13 MFCCs, shape -> (13, time_frames)
        mfcc_all = librosa.feature.mfcc(y=y_pad, sr=50, n_mfcc=13)
        # Index=1 => second MFCC coefficient, commonly referred to as 'MFCC1'
        # (since the 0th is often the energy or base coefficient).
        mfcc_coeff = mfcc_all[1, :] if mfcc_all.shape[0] > 1 else np.array([0])
        mfcc_means.append(np.mean(mfcc_coeff))
        mfcc_maxes.append(np.max(mfcc_coeff))

    df_features["mfcc1_mean"] = mfcc_means
    df_features["mfcc1_max"] = mfcc_maxes

    # Reorder the columns to match the exact 17-features from the paper
    # fmt: off
    final_order = [
        "env-impulse-factor",           # 1
        "env-margin-factor",            # 1
        "env-peak",                     # 1
        "env_freq-avg",                 # 1
        "psd-kurtosis",                 # 1
        "env-variance",                 # 6
        "1-peak-prominence-factor",     # 7
        "env-psd-entropy",              # 8
        "psd-entropy",                  # 9
        "3-peak-prominence-factor",     # 10
        "psd-skew",                     # 11
        "env-median",                   # 12
        "env-clearance-factor",         # 13
        "env-rms",                      # 14
        "freq-skew",                    # 15
        "mfcc1_mean",                   # 16
        "mfcc1_max",                    # 17
    ]
    # fmt: on
    return df_features[final_order]
