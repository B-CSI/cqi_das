"""
Copyright (c) 2025, Spanish National Research Council (CSIC)
Developed by the Barcelona Center for Subsurface Imaging (B-CSI)

This source code is subject to the terms of the
GNU Lesser General Public License.
"""

import numpy as np
import pandas as pd
import obspy
import scipy.signal
from pathlib import Path
import joblib
from joblib import Parallel, delayed

from dataclasses import dataclass
from sklearn.preprocessing import RobustScaler
from sklearn.calibration import CalibratedClassifierCV
from typing import Optional

from .signal_features import calculate_selected_features


class CQIModel:
    """
    Prediction model for Channel Quality Index (CQI).

    Loads a trained, calibrated classifier and a robust scaler to predict
    channel-quality probabilities from extracted features.
    """

    def __init__(
        self,
        clf_path: str = None,
        scaler_path: str = None,
    ) -> None:
        """
        Initialize the CQIModel with a calibrated classifier and a robust scaler.

        Parameters
        ----------
        clf_path : str, optional
            Path to the pickled calibrated classifier.
        scaler_path : str, optional
            Path to the pickled robust scaler.
        """
        # Determine the directory of this script/module
        base_path = Path(__file__).resolve().parent

        # Set default paths if not provided
        clf_path = clf_path or base_path / "trained_models/calibrated_xgb.pkl"
        scaler_path = scaler_path or base_path / "trained_models/robust_scaler.pkl"

        # Load the classifier and scaler
        self.classifier: CalibratedClassifierCV = joblib.load(clf_path)
        self.scaler: RobustScaler = joblib.load(scaler_path)

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        """
        Predict high-quality channel probabilities.

        Parameters
        ----------
        features : pd.DataFrame
            DataFrame of extracted features per channel.

        Returns
        -------
        np.ndarray
            1D array of predicted channel probabilities.
        """
        scaled_features = self.scaler.transform(features)
        return self.classifier.predict_proba(scaled_features)[:, 1]


@dataclass
class CQIPreprocessor:
    """
    Preprocessor for CQI computations.

    Applies bandpass filtering, optional decimation, and standardization,
    then extracts features.
    """

    sampling_rate: float = 50.0
    lo_pass: float = 3.0
    hi_pass: float = 20.0

    def filter_bandpass(self, data: pd.DataFrame, decimate: bool = True) -> pd.DataFrame:
        """
        Apply a bandpass filter and optional decimation to each channel.

        Parameters
        ----------
        data : pd.DataFrame
            Input signals with columns as channels and rows as samples.
        decimate : bool, optional
            Whether to decimate to 50 Hz if sampling_rate != 50.

        Returns
        -------
        pd.DataFrame
            Filtered and optionally decimated signals.
        """
        filtered_data = data.apply(
            lambda col: obspy.signal.filter.bandpass(
                np.array(col), freqmin=self.lo_pass, freqmax=self.hi_pass, df=self.sampling_rate
            )
        )

        if decimate and self.sampling_rate != 50:
            factor = int(self.sampling_rate / 50)
            filtered_data = filtered_data.apply(lambda col: scipy.signal.decimate(col, factor))

        if filtered_data.shape[0] > 6000:
            raise RuntimeWarning("Input data is longer than 2min. CQIs may be wrong")

        return filtered_data

    def remove_edges(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Remove top and bottom 5% of samples. This results in
        a time frame 10% smaller.

        Parameters
        ----------
        data : pd.DataFrame
            DataFrame with columns as channels.

        Returns
        -------
        pd.DataFrame
            Signals having top and bottom 5% of samples cut
        """

        remove_len = data.shape[0] // 20
        return data.iloc[remove_len:-remove_len,]

    def standardize(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Standardize each channel to mean 0 and std 1.

        Parameters
        ----------
        data : pd.DataFrame
            DataFrame with columns as channels.

        Returns
        -------
        pd.DataFrame
            Standardized signals.
        """
        return data.apply(lambda col: (col - col.mean()) / col.std(), axis=0)

    def calculate_features(self, scaled_data: pd.DataFrame, n_jobs: int = 1) -> pd.DataFrame:
        """
        Extract features from scaled data, optionally in parallel.

        Parameters
        ----------
        scaled_data : pd.DataFrame
            Standardized (or otherwise scaled) input where each column
            represents one channel.
        n_jobs : int, optional
            Number of parallel jobs. If 1 (default), everything is done
            sequentially. If > 1, columns are split into `n_jobs` parts
            and each part is processed in parallel.

        Returns
        -------
        pd.DataFrame
            Extracted features. The row index corresponds to the original
            channel names (i.e., columns of `scaled_data`).
        """
        # Single-process mode
        if n_jobs == 1:
            return calculate_selected_features(scaled_data)

        # Parallel mode: split columns, process each subset, then rejoin
        col_splits = np.array_split(scaled_data.columns, n_jobs)

        def _process_subset(cols):
            subset = scaled_data.loc[:, cols]
            return calculate_selected_features(subset)

        results = Parallel(n_jobs=n_jobs)(delayed(_process_subset)(cols) for cols in col_splits)

        # Concatenate along rows (i.e. channels), then reindex to restore original column order
        df_features = pd.concat(results, axis=0)
        df_features = df_features.reindex(scaled_data.columns)  # preserve channel order
        return df_features

    @staticmethod
    def sanity_check(data: pd.DataFrame) -> None:
        """
        Check if the dataset contains infty or nan.
        """
        if np.isinf(data).any(axis=None):
            raise RuntimeError("Data contains values that are infinity")

        if data.isna().any(axis=None):
            raise RuntimeError("Data contains nans")

    def size_check_plot(data: pd.DataFrame) -> None:
        pass


def calculate_cqi(
    data: pd.DataFrame,
    sampling_rate: float = 50.0,
    channel_smoothing: float = 0.5,
    decision_threshold: Optional[float] = None,
    show_plot: bool = False,
    skip_filtering: bool = False,
    skip_decimation: bool = False,
    skip_channel_norm: bool = False,
    remove_edge_effects: bool = True,
    num_jobs: int = 1,
) -> pd.Series:
    """
    Compute CQI probabilities or binary labels for each channel.

    Parameters
    ----------
    data : pd.DataFrame
        Input data with columns as channels and rows as time.
    sampling_rate : float, optional
        Sampling rate of the input signals (Hz).
    channel_smoothing : float, optional
        EMA alpha for smoothing predicted probabilities.
    decision_threshold : float, optional
        Threshold above which channels are labeled as high quality.
    show_plot : bool, optional
        If True, show an interactive threshold-adjustment plot.
    skip_filtering : bool, optional
        If True, skip bandpass filtering.
    skip_decimation : bool, optional
        If True, skip decimation.
    skip_channel_norm : bool, optional
        If True, skip standardization.
    remove_edge_effects : bool, optional
        If True, cut top and bottom 5% of time samples
    num_jobs: int, optional
        Number of jobs to run feature extraction

    Returns
    -------
    pd.Series or pd.DataFrame
        - If `decision_threshold` is provided, returns a Series of 0/1 labels.
        - Otherwise, returns channel-quality probabilities.
    """
    processor = CQIPreprocessor(sampling_rate)

    processor.sanity_check(data)

    # Apply filtering and decimation
    if not skip_filtering:
        filtered_data = processor.filter_bandpass(data, decimate=not skip_decimation)
    else:
        filtered_data = data

    # Remove edge effects (top and bottom 5% of data)
    if remove_edge_effects:
        cut_data = processor.remove_edges(filtered_data)
    else:
        cut_data = filtered_data

    # Standardize channels
    if not skip_channel_norm:
        standardized_data = processor.standardize(cut_data)
    else:
        standardized_data = cut_data

    # Extract features and predict probabilities
    features = processor.calculate_features(standardized_data, num_jobs)

    model = CQIModel()
    probabilities = model.predict(features)

    # Smooth probabilities
    smoothed_probs = pd.Series(probabilities).ewm(alpha=channel_smoothing).mean()

    # Interactive thresholding
    if show_plot:
        import matplotlib.pyplot as plt
        from .interactive_plot import create_interactive_plot

        if filtered_data.shape[0] * filtered_data.shape[1] > 5e7:
            raise RuntimeWarning("Plotting: data size may be too large to fit into memory")

        if decision_threshold is None:
            decision_threshold = 0.5

        fig, dragger = create_interactive_plot(
            cut_data, smoothed_probs, data.columns, initial_threshold=decision_threshold
        )
        plt.show()
        decision_threshold = dragger.current_threshold

    # Return labels if threshold is provided
    if decision_threshold is not None:
        return (smoothed_probs > decision_threshold).astype(int)
    return smoothed_probs
