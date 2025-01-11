import os
from argparse import ArgumentParser

import pickle

from utils import import_h5, filter_imported_data
from feature_engineering import create_feature_df

MODEL_DIR = "./results/5\ november\ results/"
MODEL_NAME = "rfe_smaller_rf.pkl"

# List with selection of features for the model
RF_SELECTED_FEATURES = [
    "dist-from-interrogator",
    "dist-from-signal-max",
    "gentle-dist-from-signal-max",
    "1-peak-prominence-factor",
    "env-skew",
    "env-margin-factor",
    "env-impulse-factor",
    "env-waveform-factor",
    "env-shape-factor",
    "env-clearance-factor",
    "mfcc1_mean",
    "env_mfcc1_mean",
    "env_mfcc1_max",
    "psd-skew",
    "env_psd-skew",
    "env-psd-entropy",
    "env_freq-skew",
    "env_freq-kurtosis",
    "env_freq-peak",
    "lags",
]


def load_model(model_name):
    with open(model_name, "rb") as f:
        predictor = pickle.load(f)
    return predictor


def load_data(filename):
    assert os.path.exists(filename), "filename not found"

    df = import_h5("data/CASTOR/z12023_06_01_06h48m33s_318.h5")
    df = filter_imported_data(
        df, pass_low=3, pass_hi=12.5, decimate_data=True, freq=100
    )
    return df


def get_predictions(data_df, predictor):
    predictor.predict()


def main(args):
    assert hasattr(args, "file"), "wrong arguments to main"

    input_df = load_data(args.file)
    features = create_feature_df(input_df)

    model_path = MODEL_DIR + MODEL_NAME
    predictor = load_model(model_path)


if __name__ == "__main__":
    parser = ArgumentParser(
        prog="Channel Quality Indicator",
        description="Assigns a quality indicator to every channel in a matrix",
    )

    # Required input file argument
    parser.add_argument(
        "-f", "--file", type=str, required=True, help="Input file in H5 format"
    )

    # Optional output file argument
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="cqi_out.csv",
        help="Output file name, defaults to 'cqi_out.csv'",
    )

    # Parse the arguments
    args = parser.parse_args()

    main(args)
