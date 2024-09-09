import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections.abc import Iterable
from matplotlib.widgets import Button
import time
import threading
import h5py


class SignalMatrixPlot:
    """
    Handle the creation of a matrix of amplitude-time plots for each
    channel. Also includes methods to update data and bad channel selection.
    """

    def __init__(self, data, dt, start_ch_index, bad_channels) -> None:
        self.data = data
        self.dt = dt
        self.bad_channels = bad_channels
        self.start_ch = start_ch_index

        # Create signal plot matrix
        self.matrix_plot_side = int(np.ceil(np.sqrt(self.data.shape[0])))
        x = self.matrix_plot_side  # make it a 10 by 10 visual matrix
        sz = 10  # matplotlib size of figure
        self.fig, self.axes = plt.subplots(nrows=x, ncols=x, figsize=(sz + sz / 2, sz))

        self._plot_per_channel_matrix()

    def update_data(self, data, start_ch_index, bad_channels):
        """Update the data displayed in the figure without closing it"""
        assert data.shape[0] <= len(self.axes.flat)
        for ax in self.axes.flat:
            ax.clear()
        self.data = data
        self.bad_channels = bad_channels
        self.start_ch = start_ch_index
        self._plot_per_channel_matrix()

    def _plot_per_channel_matrix(self):
        """Create a square matrix plot with per-channel signals."""
        data = self.data
        start_channel = self.start_ch
        num_channels = data.shape[0]

        # Make sure that axes is iterable
        if not isinstance(self.axes, Iterable):
            self.axes = np.array([self.axes])

        fig, axes = self.fig, self.axes

        for index, ax in enumerate(axes.flat):
            if index >= num_channels:
                break

            color = "C1" if self.bad_channels[index] else "C0"
            x = np.linspace(0, data.shape[1] * self.dt, data.shape[1])
            ax.plot(x, data[index, :], color=color)
            ax.title.set_text(str(index + start_channel))

        fig.tight_layout()

    def paint_signal(self, channel_idx, quality):
        """Plot an individual signal within the plot matrix signals"""
        ax = self.axes.flat[channel_idx]

        for line in ax.get_lines():
            line.remove()

        color = "C0" if quality == "good" else "C1"
        x = np.linspace(0, self.data.shape[1] * self.dt, self.data.shape[1])
        ax.plot(x, self.data[channel_idx, :], color=color)
        self.fig.canvas.draw()

    def create_zoomed_plot(self, channel_idx):
        fig, ax = plt.subplots(figsize=(10, 5))
        num_samples = len(self.data[channel_idx, :])
        x = np.linspace(0, self.dt * num_samples, num_samples)
        ax.plot(x, self.data[channel_idx, :])
        ax.set_xlabel("Time (s)")

        ch_idx = channel_idx + self.start_ch
        ax.title.set_text(f"Channel number: {ch_idx}")


class DistanceTimePlot:
    """
    Handle the creation of a matrix of the distance-time plot containing
    all channels. Also includes methods to update data and bad channel selection.
    """

    def __init__(self, data, dt, start_ch_index, bad_channels, transp=0.5) -> None:
        self.data = data
        self.dt = dt
        self.bad_channels = bad_channels
        self.start_ch = start_ch_index

        # Transparency of the black transparent line covering bad channels
        self.transparency = transp

        # Create distance-time plot
        self.fig, self.ax = plt.subplots()
        self._plot_dist_time_data()
        self._add_buttons()

    def update_data(self, data, start_ch_index, bad_channels):
        """Update the data displayed in the figure without closing it"""
        self.data = data
        self.bad_channels = bad_channels
        self.start_ch = start_ch_index
        self.ax.clear()
        self._plot_dist_time_data()

    def _add_buttons(self):
        """Add buttons in plot for interactivity"""

        # Save & Close button
        self.button_exit = self.fig.add_axes([0.76, 0.005, 0.15, 0.055])
        Button(self.button_exit, "Save & Exit")

        # Save & Next
        self.button_next = self.fig.add_axes([0.58, 0.005, 0.15, 0.055])
        Button(self.button_next, "Save & Next")

        # Select all channels
        self.button_selectall = self.fig.add_axes([0.18, 0.005, 0.15, 0.055])
        Button(self.button_selectall, "Toggle All")

    def _plot_dist_time_data(self):
        """Create distance-time plot to visualize all channels.
        Also creates an overlay for bad channels"""
        data, start_ch = self.data, self.start_ch
        cmin, cmax = -5, 5
        nd, nt = data.shape
        self.ax.imshow(
            data,
            aspect="auto",
            cmap="seismic",
            extent=[0, nt * self.dt, nd + start_ch, start_ch],
            vmin=cmin,
            vmax=cmax,
            interpolation="none",
        )
        self.ax.set_xlabel("Time (s)")
        self.ax.set_ylabel("Channel number")
        self.data_overlay = np.zeros((data.shape[0], data.shape[1], 4))
        self.data_overlay[..., 0] = 0.1  # Set channel RED
        self.data_overlay[..., 1] = 0.1  # Set channel GREEN
        self.data_overlay[..., 2] = 0.1  # Set channel BLUE
        self.overlay = self.ax.imshow(
            self.data_overlay,
            extent=[0, nt * self.dt, nd + start_ch, start_ch],
            aspect="auto",
            interpolation="none",
        )
        self.update_overlay(self.bad_channels)

    def update_overlay(self, bad_channels):
        """Re-draw the overlay of the image with the bad channels"""
        self.bad_channels = bad_channels
        self.data_overlay[..., 3] = np.tile(
            bad_channels[:, np.newaxis] * self.transparency, (1, self.data.shape[1])
        )
        self.overlay.set_data(self.data_overlay)
        self.fig.canvas.draw()


class SelectionPlotter:
    """
    DAS data plotter and manual selector of channels.

    Plots two figures:
    - Time-Distance plot.
    - Per-channel signal plot matrix.

    Selected channels are highlighted by different colors and
    have an added darker overlay.
    """

    def __init__(self, data, dt=1, start_ch_index=0, bad_channels=None) -> None:
        self.data = data
        self.dt = dt
        self.bad_channels = bad_channels
        self.start_ch = start_ch_index
        if self.bad_channels is None:
            self.bad_channels = np.zeros(data.shape[0], dtype=bool)

        # Create distance-time matrix
        self.dist_time_plt = DistanceTimePlot(
            data, dt, self.start_ch, self.bad_channels
        )

        # Create signal plot matrix
        self.matrix_plt = SignalMatrixPlot(data, dt, self.start_ch, self.bad_channels)

        # Connect the click event to the handler function
        self._enable_onclick_events()

        plt.ion()
        plt.show(block=False)
        self.closed = False

        # To handle double clicks
        self.single_click_event = None

    def update_data(self, data, start_ch_index, bad_channels):
        """Update the displayed data without closing the figures"""
        self.data = data
        self.bad_channels = bad_channels
        self.start_ch = start_ch_index

        self.dist_time_plt.update_data(data, start_ch_index, bad_channels)
        self.matrix_plt.update_data(data, start_ch_index, bad_channels)

    def paint_channel(self, channel_idx, quality):
        """Convert channel to good or bad and paint plots"""
        assert quality in ["good", "bad"]

        # Update quality for channel idx
        changed = False
        if quality == "good" and self.bad_channels[channel_idx]:
            self.bad_channels[channel_idx] = False
            changed = True
        elif quality == "bad" and not self.bad_channels[channel_idx]:
            self.bad_channels[channel_idx] = True
            changed = True

        if not changed:
            return

        # Draw changes in the plots
        self.matrix_plt.paint_signal(channel_idx, quality)
        self.dist_time_plt.update_overlay(self.bad_channels)

    def _on_click_in_dist_time_plot(self, event):
        """Handle clicks in the channel vs time plot."""
        if event.inaxes is self.dist_time_plt.ax:
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

        elif event.inaxes is self.dist_time_plt.button_exit:
            self._on_click_button_close(event)

        elif event.inaxes is self.dist_time_plt.button_next:
            self._on_click_button_next(event)

        elif event.inaxes is self.dist_time_plt.button_selectall:
            self._on_click_button_select_all(event)

    def _on_click_in_plot_matrix(self, event):
        """Handle clicks in the per-channel signal matrix plot."""
        for channel_idx in range(len(self.data)):
            # Find the axes that was clicked on
            if self.matrix_plt.axes.flat[channel_idx] == event.inaxes:
                # On double-click event
                if event.dblclick:
                    # Cancel a single click event triggered by the dblclick
                    if self.single_click_event is not None:
                        self.single_click_event.cancel()
                        self.single_click_event = None

                    # Create a zoomed plot with the clicked channel data
                    self.matrix_plt.create_zoomed_plot(channel_idx)
                    return
                else:
                    # Wait for other single click events to terminate
                    if self.single_click_event is not None:
                        while self.single_click_event.is_alive():
                            time.sleep(0.1)
                        self.single_click_event = None

                    if event.button == 1:
                        click_type = "bad"
                    elif event.button == 3:
                        click_type = "good"

                    # Launch event on a timer to account for double clicks
                    self.single_click_event = threading.Timer(
                        0.1, self.paint_channel, [channel_idx, click_type]
                    )
                    self.single_click_event.start()
                    return

    def _on_click_button_next(self, _):
        self.dist_time_plt.fig.canvas.stop_event_loop()

    def _on_click_button_close(self, _):
        self._on_close_event()
        self.dist_time_plt.fig.canvas.stop_event_loop()

    def _on_click_button_select_all(self, _):
        """Select all channels, or deselect if already selected"""
        bad_channels = np.ones(self.bad_channels.shape)
        if self.bad_channels.all():
            bad_channels = np.zeros(self.bad_channels.shape)

        self.update_data(self.data, self.start_ch, bad_channels)

    def _on_close_event(self, _=None):
        plt.close(self.dist_time_plt.fig)
        plt.close(self.matrix_plt.fig)
        self.closed = True

    def _enable_onclick_events(self):
        """Connect and enable the click events in the figures"""
        self.dist_time_plt.fig.canvas.mpl_connect(
            "button_press_event", self._on_click_in_dist_time_plot
        )
        self.matrix_plt.fig.canvas.mpl_connect(
            "button_press_event", self._on_click_in_plot_matrix
        )

        # When figures are closed
        self.dist_time_plt.fig.canvas.mpl_connect("close_event", self._on_close_event)
        self.matrix_plt.fig.canvas.mpl_connect("close_event", self._on_close_event)

    def get_selections(self):
        """Get bad_channel selections"""
        return self.bad_channels.astype(int)

    def start(self):
        self.dist_time_plt.fig.canvas.start_event_loop(timeout=-1)

    def close(self):
        self._on_close_event()


class ChannelSelector:
    """
    Channel Quality Selector.
    It uses a SelectionPlotter to draw the figures and record the selections.
    ChannelSelector handles the data slicing and saves the data.
    It includes automatic selection using SNR.

    Args:
    - data: [distance, time]
    - dt: time step (delta)
    - start_channel: number assigned to the first channel that appears in data
    - output_fname: name of the CSV file where to store the selections.
    """

    def __init__(self, data, dt=1, start_channel=0, output_fname=None):
        self.data = data
        self.dt = dt
        self.oname = output_fname
        self.start_channel = start_channel
        self.selections = np.zeros(len(data), dtype=int)

    @classmethod
    def read_from_csv(cls, input_filename, dt=1):
        data = pd.read_csv(input_filename, sep=",", index_col=0)
        return cls(data, dt)

    def auto_select_using_snr(self, threshold=5):
        """Find bad channels automatically according to a criterion."""

        def rms(values):
            return np.sqrt(np.mean(np.square(values)))

        df_features = pd.DataFrame()
        df_features["rms"] = np.apply_along_axis(rms, 1, self.data)
        df_features["peak"] = np.abs(self.data).max(axis=1)
        df_features["crest-factor"] = df_features["peak"] / df_features["rms"]

        bad_channels = np.where(df_features["crest-factor"] < threshold, 0, 0)
        self.selections = bad_channels

    def select(self, slice_size=25, channel_shift=0, last_ch=None):
        if last_ch is None:
            last_ch = len(self.selections)
        assert last_ch <= len(self.selections), "Last channel is larger than available"
        assert (
            channel_shift < last_ch
        ), "First channel must be smaller than last channel"

        first_ch = channel_shift + self.start_channel
        plotter = SelectionPlotter(self.data[:slice_size, :], self.dt, first_ch)

        for idx in range(0, self.data.shape[0], slice_size):
            if plotter.closed:
                print("Channel selector closed at index:", idx - 1)
                break
            data_slice = self.data[idx : idx + slice_size, :]
            start_ch = idx + first_ch
            bad_channels_slice = self.selections[idx : idx + slice_size]
            plotter.update_data(data_slice, start_ch, bad_channels_slice)
            plotter.start()

            manual_selections = plotter.get_selections()
            print(manual_selections)
            self.selections[idx : idx + slice_size] = manual_selections
            self.save_selections()

        plotter.close()

    def get_selections(self, first_ch=None, last_ch=None):
        """Return a numpy array with the selections."""
        if last_ch is None:
            last_ch = len(self.selections) + self.start_channel
        assert (
            last_ch <= len(self.selections) + self.start_channel
        ), "Last channel is larger than available"
        if first_ch is None:
            first_ch = self.start_channel
        assert first_ch < last_ch, "First channel must be smaller than last channel"
        assert (
            first_ch >= self.start_channel
        ), "The provided first channel is smaller than the first channel in the data"

        # Create the DataFrame with the custom index
        index = range(first_ch, last_ch)
        data = self.selections[
            first_ch - self.start_channel : last_ch - self.start_channel
        ]
        df = pd.DataFrame(data, index=index, columns=["Quality"])

        return df

    def save_selections(self, output_fname=None):
        """Write out the selections to the output filename"""
        if self.oname is None and output_fname is None:
            return

        if output_fname is None:
            output_fname = self.oname

        selections = self.get_selections()
        selections.to_csv(output_fname, sep=",")


if __name__ == "__main__":
    fpath = "data/hdas/20230720-114924.h5"
    filename = "data/hdas/20230720-114924.h5"
    with h5py.File(filename, "r") as f:
        data = f["data"][:]
        dt = f["data"].attrs["dt_s"][()]

    data = data[1000:1100, 1600:2000]

    # This is the basic usage
    start_channel = 0
    selector = ChannelSelector(data, dt, start_channel, output_fname="quality_mark.csv")
    selector.auto_select_using_snr(threshold=3)
    selector.select(slice_size=49, channel_shift=0)

    print(selector.get_selections())
