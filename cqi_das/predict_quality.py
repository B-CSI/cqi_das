import numpy as np
import pandas as pd
import obspy
import joblib
import scipy.signal

from dataclasses import dataclass
from sklearn.preprocessing import RobustScaler
from sklearn.calibration import CalibratedClassifierCV
from typing import Optional

from signal_features import calculate_selected_features


class CQIModel:
    """
    Prediction model for Channel Quality Index (CQI).

    Loads a trained, calibrated classifier and a robust scaler to predict
    channel-quality probabilities from extracted features.
    """

    def __init__(
        self,
        clf_path: str = "./trained_models/calibrated_xgb.pkl",
        scaler_path: str = "./trained_models/robust_scaler.pkl",
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

        return filtered_data

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

    def calculate_features(self, scaled_data: pd.DataFrame) -> pd.DataFrame:
        """
        Extract features from scaled data.

        Parameters
        ----------
        scaled_data : pd.DataFrame
            Standardized (or otherwise scaled) input.

        Returns
        -------
        pd.DataFrame
            Extracted features.
        """
        return calculate_selected_features(scaled_data)


def calculate_cqi(
    data: pd.DataFrame,
    sampling_rate: float = 50.0,
    channel_smoothing: float = 0.5,
    decision_threshold: Optional[float] = None,
    interactive: bool = False,
    skip_filtering: bool = False,
    skip_decimation: bool = False,
    skip_channel_norm: bool = False,
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
    interactive : bool, optional
        If True, show an interactive threshold-adjustment plot.
    skip_filtering : bool, optional
        If True, skip bandpass filtering.
    skip_decimation : bool, optional
        If True, skip decimation.
    skip_channel_norm : bool, optional
        If True, skip standardization.

    Returns
    -------
    pd.Series or pd.DataFrame
        - If `decision_threshold` is provided, returns a Series of 0/1 labels.
        - Otherwise, returns channel-quality probabilities.
    """
    processor = CQIPreprocessor(sampling_rate)

    # Apply filtering and decimation
    if not skip_filtering:
        filtered_data = processor.filter_bandpass(data, decimate=not skip_decimation)
    else:
        filtered_data = data

    # Standardize channels
    if not skip_channel_norm:
        standardized_data = processor.standardize(filtered_data)
    else:
        standardized_data = filtered_data

    # Extract features and predict probabilities
    features = processor.calculate_features(standardized_data)
    model = CQIModel()
    probabilities = model.predict(features)

    # Smooth probabilities
    smoothed_probs = pd.Series(probabilities).ewm(alpha=channel_smoothing).mean()

    # Interactive thresholding
    if interactive:
        import matplotlib.pyplot as plt
        from interactive_plot import create_interactive_plot

        if decision_threshold is None:
            decision_threshold = 0.5

        fig, dragger = create_interactive_plot(
            data, smoothed_probs, data.columns, initial_threshold=decision_threshold
        )
        plt.show()
        decision_threshold = dragger.current_threshold

    # Return labels if threshold is provided
    if decision_threshold is not None:
        return (smoothed_probs > decision_threshold).astype(int)
    return smoothed_probs
