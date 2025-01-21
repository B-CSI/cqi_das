from cqi_das import utils, calculate_cqi

# candas222_df = utils.import_h5("data/CANDAS2/CANDAS2_2023-02-22_07-01-14_filtered_2pn.h5")

# candas222_df = utils.set_data_limits(candas222_df, first_ch=415, first_time=3000, last_time=5500)

candas2_df = utils.import_h5("data/CANDAS2/CANDAS2_2022-12-27_07-46-15.h5")

# candas2_df = utils.filter_imported_data(candas2_df, pass_low=3, pass_hi=20, decimate_data=True)

candas2_df = utils.set_data_limits(candas2_df, first_ch=450, first_time=2500, last_time=6500)

plot_parameters = {
    "save_path": "./figure_test.png",
    "figure_size": (10, 10),
    "vmin": -5,
    "vmax": 5,
}

cqis = calculate_cqi(
    candas2_df,
    sampling_rate=50,
    channel_smoothing=0.5,
    show_plot=True,
    num_jobs=8,
    plot_parameters=plot_parameters,
)
print(cqis)
