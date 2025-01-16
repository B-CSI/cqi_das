from cqi_das import utils, calculate_cqi

candas222_df = utils.import_h5("data/CANDAS2/CANDAS2_2023-02-22_07-01-14_filtered_2pn.h5")

candas222_df = utils.set_data_limits(candas222_df, first_ch=415, first_time=3000, last_time=5500)

cqis = calculate_cqi(
    candas222_df, sampling_rate=50, channel_smoothing=0.5, interactive=True, num_jobs=4
)
print(cqis)
