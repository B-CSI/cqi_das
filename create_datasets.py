#!/usr/bin/env python

import pandas as pd
import numpy as np
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed

# Local imports
import utils
from feature_engineering import create_feature_df, run_pcc


def standardize_channels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize each channel (column) independently to mean=0, std=1.
    """
    df_standardized = df.apply(lambda col: (col - col.mean()) / col.std(), axis=0)
    return df_standardized


def process_dataset(df: pd.DataFrame, bad_channels: np.ndarray, event_label: str) -> pd.DataFrame:
    """
    1) Standardize each channel of the input DataFrame.
    2) Extract features.
    3) Run PCC on a subset (or all) of the data and create 'beta' features.
    4) Mark bad channels with target=1, others=0.
    5) Save the final feature DataFrame to CSV.

    :param df:            Time x Channels DataFrame
    :param bad_channels:  1D array/list of channel indices that are "bad"
    :param event_label:   String for labeling/saving results

    :return:              DataFrame of features with an added 'target' column
    """

    # 1) Standardize each channel
    df_std = standardize_channels(df)

    # 2) Extract initial features
    df_features = create_feature_df(df_std)

    df_for_pcc = df_std

    pcc_results = run_pcc(df_for_pcc, event_date=event_label)

    # Add PCC-based features to df_features
    df_features["beta"] = pcc_results["pcc_feature"]
    df_features["modified-beta"] = pcc_results["modified_pcc_feature"]
    df_features["mean_pcc"] = pcc_results["pcc_mean_feature"]
    df_features["median_pcc"] = pcc_results["pcc_median_feature"]
    df_features["mad_pcc"] = pcc_results["pcc_mad_feature"]
    df_features["lags"] = pcc_results["pcc_lags_feature"]

    # 4) Create 'target' column
    df_features["target"] = 0

    # Make sure index is int if your channels are int-based
    df_features.index = df_features.index.astype(int)
    # Also ensure bad_channels is int-based
    bad_channels = [int(ch) for ch in bad_channels]

    # Intersection
    common_ch = df_features.index.intersection(bad_channels)
    df_features.loc[common_ch, "target"] = 1

    # Optional: warn about missing channels
    missing = set(bad_channels) - set(df_features.index)
    if missing:
        print(f"[{event_label}] Warning: The following channels are not found: {missing}")

    # 5) Save DataFrame
    outfile = f"{event_label}_features.csv"
    df_features.to_csv(outfile)
    print(f"[{event_label}] Saved features (including target) to {outfile}")

    return df_features


def process_one_dataset(config: dict) -> None:
    """
    Worker function to handle loading, filtering, limiting,
    bad channels, and calling process_dataset().
    Wrapped in try/except to avoid interrupting other tasks if it fails.
    """

    name = config["name"]
    try:
        print(f"[{name}] Starting processing...")

        # 1) Load data
        if config.get("use_miniseed", False):
            df = utils.import_miniseed(config["pathname"], config["year"], config["exp_abbr"])
        else:
            df = utils.import_h5(config["pathname"])

        # 2) Filter, decimate, etc.
        df = utils.filter_imported_data(
            df,
            pass_low=config.get("pass_low", 3),
            pass_hi=config.get("pass_hi", 20),
            freq=config.get("freq", 50),
            decimate_data=config.get("decimate_data", True),
        )

        # 3) Set data limits
        df = utils.set_data_limits(
            df,
            first_ch=config.get("first_ch"),
            last_ch=config.get("last_ch"),
            first_time=config.get("first_time"),
            last_time=config.get("last_time"),
        )

        # 4) Load bad channels
        if config.get("candas1", False):
            bad_channels = utils.load_bad_channels(candas1=True)
        else:
            bad_channels = utils.load_bad_channels(
                filename=config["bad_channels_path"], first_ch=config["first_ch"]
            )

        # 5) Process dataset (will create CSV)
        process_dataset(
            df=df,
            bad_channels=bad_channels,
            event_label=name,
        )

        print(f"[{name}] Finished without errors.")

    except Exception as e:
        print(f"[{name}] FAILED: {str(e)}")
        traceback.print_exc()


def main():
    # A list of dictionaries defining each dataset
    dataset_configs = [
        {
            "name": "candas2",
            "pathname": "data/CANDAS2/CANDAS2_2022-12-27_07-46-15.h5",
            "bad_channels_path": "channel_selections/quality_picks_candas2_picked_0730_1940.csv",
            "pass_low": 3,
            "pass_hi": 20,
            "freq": 50,
            "decimate_data": True,
            "first_ch": 415,
            "first_time": 3000,
            "last_time": 6000,
        },
        {
            "name": "safe_t",
            "pathname": "data/SAFE/ICM_2023-11-13_14-57-38.h5",
            "bad_channels_path": "channel_selections/CQI_selections_SAFE_2023-11-13_14-57-38_picked_1021_2043.csv",
            "pass_low": 3,
            "pass_hi": 20,
            "freq": 100,
            "decimate_data": True,
            "first_ch": 236,
            "first_time": 3000,
            "last_time": 6500,
        },
        {
            "name": "castor2",
            "pathname": "data/CASTOR/z12023_06_01_06h48m33s_318.h5",
            "bad_channels_path": "channel_selections/CQI_selections_CASTOR_2023_06_01_picked_1004_1611.csv",
            "pass_low": 3,
            "pass_hi": 20,
            "freq": 100,
            "decimate_data": True,
            "first_ch": 1082,
            "first_time": 2500,
            "last_time": 4000,
        },
        {
            "name": "twin_gc",
            "pathname": "data/CANDAS/GC/ZI.G20200822_1713.h5",
            "bad_channels_path": "channel_selections/quality_picks_gc_twin_event_G20200822_1713_picked_0802_1839.csv",
            "pass_low": 3,
            "pass_hi": 20,
            "decimate_data": True,
            "first_ch": 415,
            "last_time": 3000,
        },
        {
            "name": "twin_tf",
            "pathname": "data/CANDAS/TF/ZI.T20200822_1713.h5",
            "bad_channels_path": "channel_selections/quality_picks_tf_twin_event_T20200822_1713_picked_0807_0337.csv",
            "pass_low": 3,
            "pass_hi": 20,
            "decimate_data": True,
            "first_ch": 745,
            "first_time": 4000,
            "last_time": 6000,
        },
        {
            "name": "tf_2nd_pick",
            "pathname": "data/CANDAS/TF/ZI.T20200727_2044.h5",
            "bad_channels_path": "channel_selections/CQI_selections_T20200727_2044_picked_0912_1800.csv",
            "pass_low": 3,
            "pass_hi": 20,
            "decimate_data": True,
            "first_ch": 745,
            "first_time": 4000,
        },
        {
            "name": "tf_same_day",
            "pathname": "data/CANDAS/TF/ZI.T20200727_1305.h5",
            "bad_channels_path": "channel_selections/CQI_selections_T20200727_1305_picked_0918_1813.csv",
            "pass_low": 3,
            "pass_hi": 20,
            "decimate_data": True,
            "first_ch": 745,
            "first_time": 4500,
            "last_time": 7500,
        },
        {
            "name": "castor",
            "pathname": "data/CASTOR/20230720-115624.h5",
            "bad_channels_path": "channel_selections/quality_picks_castor2_20230720_picked_0813_2235.csv",
            "pass_low": 3,
            "pass_hi": 20,
            "freq": 100,
            "decimate_data": True,
            "first_ch": 1082,
            "first_time": 1800,
            "last_time": 2800,
        },
        {
            "name": "candas1",
            "use_miniseed": True,  # <--- This signals we should use import_miniseed
            "pathname": "data/CANDAS/GC/20200815_19h05m/eventCANDAS/",
            "year": 2020,
            "exp_abbr": "G",
            "bad_channels_path": None,  # not used if candas1=True
            "candas1": True,  # triggers special load in utils.load_bad_channels
            "pass_low": 3,
            "pass_hi": 20,
            "freq": 50,
            "decimate_data": True,
            "first_ch": 415,
            "first_time": 13000,
            "last_time": 18000,
        },
        {
            "name": "safe2",
            "pathname": "data/SAFE/ICM_2023-12-15_23-43-29.h5",
            "bad_channels_path": "channel_selections/CQI_selections_SAFE_2023-12-15_23-43-29_picked_1203.csv",
            "pass_low": 3,
            "pass_hi": 20,
            "freq": 100,
            "decimate_data": True,
            "first_ch": 236,
            "first_time": 5000,
            "last_time": 7500,
        },
    ]

    dataset_configs = dataset_configs[-1:]
    print(dataset_configs)

    max_workers = 4  # Adjust based on how many cores you want to use
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_one_dataset, config) for config in dataset_configs]

        # We collect results in a simple loop so that if one fails, others continue
        for future in as_completed(futures):
            # Each future may raise an exception if the dataset fails
            # but we've also wrapped the logic inside try/except in process_one_dataset
            # so typically we won't see an exception here unless it's a deeper-level issue.
            try:
                result = future.result()  # will be None if everything is handled
            except Exception as e:
                print(f"[MAIN] Caught unexpected top-level error: {e}")

    print("\nAll dataset submissions have completed.")
    print("Check the logs above for any dataset that might have failed.")


if __name__ == "__main__":
    main()
