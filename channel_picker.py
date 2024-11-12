import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections.abc import Iterable
from matplotlib.widgets import Button
import time
import h5py
import utils as utl
from pathlib import Path
from datetime import datetime


class PickPlotter:
    """
    DAS data plotter and manual picker of channels.

    Plots two figures:
    - Time-Distance plot.
    - Per-channel signal plot matrix.

    Selected channels are highlighted by different colors and
    have an added darker overlay.
    """

    def __init__(
        self, data, dt=1, start_ch_index=0, bad_channels=None, transparency=0.5
    ) -> None:
        self.data = data
        self.dt = dt
        self.bad_channels = bad_channels
        self.start_ch = start_ch_index
        if self.bad_channels is None:
            self.bad_channels = np.zeros(data.shape[0], dtype=bool)

        # Transparency of the black transparent line covering bad channels
        self.transparency = transparency

        # Create distance-time plot
        self.fig_dist_time, self.ax_dist_time = plt.subplots()
        self._plot_dist_time_data(data, start_ch_index)

        # Create signal plot matrix
        self.fig_matrix, self.ax_matrix = None, None
        self._plot_per_channel_matrix(data, start_ch_index)

        # Connect the click event to the handler function
        self._enable_onclick_events()

        plt.ion()
        plt.show(block=True)

    def _plot_dist_time_data(self, data, start_ch):
        """Create distance-time plot to visualize all channels.
        Also creates an overlay for bad channels"""
        cmin, cmax = -5, 5
        nd, nt = data.shape
        self.ax_dist_time.imshow(
            data,
            aspect="auto",
            cmap="seismic",
            extent=[0, nt * self.dt, nd + start_ch, start_ch],
            vmin=cmin,
            vmax=cmax,
            interpolation="none",
        )
        self.ax_dist_time.set_xlabel("Time (s)")
        self.ax_dist_time.set_ylabel("Channel number")
        self.data_overlay = np.zeros((data.shape[0], data.shape[1], 4))
        self.data_overlay[..., 0] = 0.1  # Set channel RED
        self.data_overlay[..., 1] = 0.1  # Set channel GREEN
        self.data_overlay[..., 2] = 0.1  # Set channel BLUE
        self.overlay = self.ax_dist_time.imshow(
            self.data_overlay,
            extent=[0, nt * self.dt, nd + start_ch, start_ch],
            aspect="auto",
            interpolation="none",
        )

    def _plot_per_channel_matrix(self, data, start_channel=0):
        """Create a square matrix plot with per-channel signals."""
        num_channels = data.shape[0]

        self.matrix_plot_side = int(np.ceil(np.sqrt(self.data.shape[0])))
        x = self.matrix_plot_side  # make it a 10 by 10 visual matrix
        sz = 10  # matplotlib size of figure
        fig, axes = plt.subplots(nrows=x, ncols=x, figsize=(sz + sz / 2, sz))

        # Make sure that axes is iterable
        if not isinstance(axes, Iterable):
            axes = np.array([axes])

        for index, ax in enumerate(axes.flat):
            if index >= num_channels:
                break

            color = "C1" if self.bad_channels[index] else "C0"
            x = np.linspace(0, data.shape[1] * self.dt, data.shape[1])
            ax.plot(x, data[index, :], color=color)
            ax.set_title(str(index + start_channel), fontsize=10)

        fig.tight_layout()
        fig.subplots_adjust(wspace=0.15, hspace=0.15) #if want to cram as many subplots as possible together
        self.fig_matrix, self.ax_matrix = fig, axes

    def _update_overlay(self, bad_channels):
        """Re-draw the overlay of the image with the bad channels"""
        self.data_overlay[..., 3] = np.tile(
            bad_channels[:, np.newaxis] * self.transparency, (1, self.data.shape[1])
        )
        self.overlay.set_data(self.data_overlay)
        self.fig_dist_time.canvas.draw()

    def _paint_signal(self, channel_idx, quality):
        """Plot an individual signal within the plot matrix signals"""
        ax = self.ax_matrix.flat[channel_idx]

        for line in ax.get_lines():
            line.remove()

        color = "C0" if quality == "good" else "C1"
        x = np.linspace(0, self.data.shape[1] * self.dt, self.data.shape[1])
        ax.plot(x, self.data[channel_idx, :], color=color)
        self.fig_matrix.canvas.draw()

    def paint_channel(self, channel_idx, quality):
        """Convert channel to good or bad and paint plots"""
        assert quality in ["good", "bad"]
        changed = False
        if quality == "good" and self.bad_channels[channel_idx]:
            self.bad_channels[channel_idx] = False
            changed = True
        elif quality == "bad" and not self.bad_channels[channel_idx]:
            self.bad_channels[channel_idx] = True
            changed = True

        if not changed:
            return

        self._paint_signal(channel_idx, quality)
        self._update_overlay(self.bad_channels)

    def _on_click_in_dist_time_plot(self, event):
        """Handle clicks in the channel vs time plot."""
        if event.inaxes is self.ax_dist_time:
            row = int(event.ydata)
            col = int(event.xdata)
            channel = row - self.start_ch

            if event.button == 1:  # Left click
                click_type = "bad"
            elif event.button == 3:  # Right click
                click_type = "good"
            else:
                return

            self.paint_channel(channel, click_type)
        elif event.inaxes is self.axbutton:
            # Clicked the button
            self._on_click_button(event)

    def _on_click_in_plot_matrix(self, event):
        """Handle clicks in the per-channel signal matrix plot."""
        for channel_idx in range(len(self.data)):
            # Find the axes that was clicked on
            if self.ax_matrix.flat[channel_idx] == event.inaxes:
                if event.button == 1:
                    click_type = "bad"
                elif event.button == 3:
                    click_type = "good"

                self.paint_channel(channel_idx, click_type)
                return

    def _on_click_button(self, _):
        self._on_close_event(None)

    def _on_close_event(self, _):
        plt.close(self.fig_dist_time)
        plt.close(self.fig_matrix)

    def _enable_onclick_events(self):
        self.fig_dist_time.canvas.mpl_connect(
            "button_press_event", self._on_click_in_dist_time_plot
        )
        self.fig_matrix.canvas.mpl_connect(
            "button_press_event", self._on_click_in_plot_matrix
        )

        # When figures are closed
        self.fig_dist_time.canvas.mpl_connect("close_event", self._on_close_event)
        self.fig_matrix.canvas.mpl_connect("close_event", self._on_close_event)

        # Save button
        self.axbutton = self.fig_dist_time.add_axes([0.86, 0.005, 0.1, 0.055])
        bsave = Button(self.axbutton, "Close")
        bsave.on_clicked(self._on_click_button)

    def get_picks(self):
        """Get bad_channel_picks"""
        return self.bad_channels.astype(int)


class ChannelPicker:
    """
    Channel Quality Picker.
    It uses a PickPlotter to draw the figures and record the picks.
    ChannelPicker handles the data slicing and saves the data.
    It includes automatic picking using SNR.

    Args:
    - data: [distance, time]
    - dt: time step (delta)
    - start_channel: number assigned to the first channel that appears in data
    - output_fname: name of the CSV file where to store the picks.
    """

    def __init__(self, data, dt=1, start_channel=0, output_fname=None):
        self.data = data
        self.dt = dt
        self.oname = output_fname
        self.start_channel = start_channel
        self.picks = np.zeros(len(data), dtype=int)

    @classmethod
    def read_from_csv(cls, input_filename, dt=1):
        data = pd.read_csv(input_filename, sep=",", index_col=0)
        return cls(data, dt)

    def autopick_using_snr(self, threshold=5):
        """Find bad channels automatically according to a criterion."""

        def rms(values):
            return np.sqrt(np.mean(np.square(values)))

        df_features = pd.DataFrame()
        df_features["rms"] = np.apply_along_axis(rms, 1, self.data)
        df_features["peak"] = np.abs(self.data).max(axis=1)
        df_features["crest-factor"] = df_features["peak"] / df_features["rms"]

        bad_channels = np.where(df_features["crest-factor"] < threshold, 0, 0)
        self.picks = bad_channels

    def autopick_using_root_amplitude(self, threshold=5):
        """Find bad channels automatically according to a criterion."""

        def root_amplitude(values):
            return np.square(np.mean(np.sqrt(np.abs(values))))
        
        df_features = pd.DataFrame()
        df_features['root-amplitude'] = np.apply_along_axis(root_amplitude, 1, self.data)

        bad_channels = np.where(df_features["root-amplitude"] > threshold, 0, 0)
        self.picks = bad_channels

    def pick(self, slice_size=25, channel_shift=0, last_ch=None):
        if last_ch is None:
            last_ch = len(self.picks)
        assert last_ch <= len(self.picks), "Last channel is larger than available"
        assert (
            channel_shift < last_ch
        ), "First channel must be smaller than last channel"

        for idx in range(0, self.data.shape[0], slice_size):
            data_slice = self.data[idx : idx + slice_size, :]
            start_ch = idx + channel_shift + self.start_channel
            bad_channels_slice = self.picks[idx : idx + slice_size]
            plotter = PickPlotter(data_slice, self.dt, start_ch, bad_channels_slice)

            manual_picks = plotter.get_picks()
            print(manual_picks)
            self.picks[idx : idx + slice_size] = manual_picks
            self.save_picks()

    def get_picks(self, first_ch=0, last_ch=None):
        """Return a numpy array with the picks."""
        if last_ch is None:
            last_ch = len(self.picks)
        assert last_ch <= len(self.picks), "Last channel is larger than available"
        assert first_ch < last_ch, "First channel must be smaller than last channel"

        # Create the DataFrame with the custom index
        index = range(first_ch, last_ch)
        data = self.picks[first_ch:last_ch]
        df = pd.DataFrame(data, index=index, columns=["Quality"])

        return df

    def save_picks(self, output_fname=None):
        """Write out the picks to the output filename"""
        if self.oname is None and output_fname is None:
            return

        if output_fname is None:
            output_fname = self.oname

        picks = self.get_picks()
        picks.to_csv(output_fname, sep=",")


if __name__ == "__main__":
    #fpath = "data/hdas/20230720-114924.h5"
    #filename = "data/CANDAS/TF/ZI.T20200727_2044.h5"
    #filename = '/home/tatiana/Documents/Onboarding_to_TREMORS/CANDAS2/CANDAS2_2022-12-27_07-46-15.h5'
    #gc_file = 'data/toTatiana/events_CANDAS_TF&GC/ZI.G20200822_1713.h5' 
    #filepath = 'data/toTatiana/events_CANDAS_TF&GC/ZI.T20200822_1713.h5'
    #filepath = 'data/CASTOR/20230720-114924.h5' 
    filepath = 'data/CASTOR/20230720-115624.h5'

    #filepath = '/home/tatiana/Documents/Onboarding_to_TREMORS/CANDAS/GC/20200815_19h05m/eventCANDAS/'
    #data = utl.import_miniseed(filepath, 2020, 'G')
    
    #data = utl.import_h5(gc_file)
    data = utl.import_h5(filepath)
    freq = 100
    data = utl.filter_imported_data(data,pass_low=3, pass_hi=12.5, freq=freq, decimate_data=False)
    #data = utl.set_data_limits(data, first_ch=415, first_time=50, last_time=2500) #gc
    first_ch = 1082
    data = utl.set_data_limits(data, first_ch=first_ch, first_time=3500, last_time=5500)

    # last_ch = 8000
    # first_time = 3500
    # last_time = 4500
    # data = utl.set_data_limits(data, first_ch=first_ch, last_ch=last_ch, first_time=first_time, last_time=last_time)
    data = data.T.to_numpy()

    # This is the basic usage
    picker = ChannelPicker(data, 1./freq, first_ch, output_fname="quality_picks.csv")
    picker.autopick_using_snr(threshold=4.5)
    #picker.autopick_using_root_amplitude(threshold=3)
    picker.pick(slice_size=36, channel_shift=0) #for saving channels

    print(picker.get_picks())
    timestamp = datetime.now().strftime("%m%d_%H%M")
    if filepath.endswith(".h5"):
        picker.save_picks('quality_picks_' + filepath.split('/')[-1].split('.')[1] + '_picked_' + timestamp + '.csv')
    else:
        picker.save_picks('quality_picks_' + "miniseed_RENAME_ME" + '_' + timestamp + '.csv')