"""
Copyright (c) 2025, Spanish National Research Council (CSIC)
Developed by the Barcelona Center for Subsurface Imaging (B-CSI)

This source code is subject to the terms of the
GNU Lesser General Public License.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections.abc import Iterable
from matplotlib.widgets import Button
import time
import threading
import utils as utl
from datetime import datetime


class SignalMatrixPlot:
    """
    Creates and manages a grid (matrix) of subplots for displaying
    individual channel signals. Each subplot corresponds to one channel.

    Parameters
    ----------
    data : np.ndarray
        2D NumPy array of shape (n_channels, n_samples).
    dt : float
        Time step (delta) between samples.
    start_ch_index : int
        The index of the first channel in `data` (used mainly for labeling).
    bad_channels : np.ndarray
        Boolean array indicating which channels are currently marked as "bad".
    """

    def __init__(
        self, data: np.ndarray, dt: float, start_ch_index: int, bad_channels: np.ndarray
    ) -> None:
        self.data = data
        self.dt = dt
        self.bad_channels = bad_channels
        self.start_ch = start_ch_index

        # Determine how many subplots are needed on each side
        # (e.g., if 10 channels, create a 4x4 grid to fit them).
        self.matrix_plot_side = int(np.ceil(np.sqrt(self.data.shape[0])))
        x = self.matrix_plot_side

        # Create a figure with x-by-x subplots
        sz = 10
        self.fig, self.axes = plt.subplots(nrows=x, ncols=x, figsize=(sz + sz / 2, sz))

        # Plot the signals for each channel initially
        self._plot_per_channel_matrix()

    def update_data(self, data: np.ndarray, start_ch_index: int, bad_channels: np.ndarray) -> None:
        """
        Update the existing subplots with new data, allowing the figure to persist.

        Parameters
        ----------
        data : np.ndarray
            New data array of shape (n_channels, n_samples).
        start_ch_index : int
            Index of the first channel for labeling.
        bad_channels : np.ndarray
            Boolean array indicating which channels are "bad".
        """
        # Clear old plots from all subplots
        assert data.shape[0] <= len(self.axes.flat)
        for ax in self.axes.flat:
            ax.clear()

        self.data = data
        self.bad_channels = bad_channels
        self.start_ch = start_ch_index
        self._plot_per_channel_matrix()

    def _plot_per_channel_matrix(self) -> None:
        """
        Populate the subplots with one signal per channel.
        Color code indicates whether a channel is "bad" or "good".
        """
        data = self.data
        start_channel = self.start_ch
        num_channels = data.shape[0]

        # Ensure self.axes is iterable (it could be a single Axes object if only 1x1).
        if not isinstance(self.axes, Iterable):
            self.axes = np.array([self.axes])

        fig, axes = self.fig, self.axes

        for index, ax in enumerate(axes.flat):
            if index >= num_channels:
                break

            # Choose color based on bad/good status of the channel
            color = "C1" if self.bad_channels[index] else "C0"
            x_vals = np.linspace(0, data.shape[1] * self.dt, data.shape[1])
            ax.plot(x_vals, data[index, :], color=color)
            ax.set_title(str(index + start_channel))

        fig.tight_layout()

    def paint_signal(self, channel_idx: int, quality: str) -> None:
        """
        Re-draw a single channel's subplot in a new color to reflect its
        'good' or 'bad' status.

        Parameters
        ----------
        channel_idx : int
            Which channel subplot to repaint.
        quality : str
            'good' or 'bad', to determine which color to use.
        """
        ax = self.axes.flat[channel_idx]

        # Remove the existing line object before drawing a new one
        for line in ax.get_lines():
            line.remove()

        color = "C0" if quality == "good" else "C1"
        x_vals = np.linspace(0, self.data.shape[1] * self.dt, self.data.shape[1])
        ax.plot(x_vals, self.data[channel_idx, :], color=color)
        self.fig.canvas.draw()

    def create_zoomed_plot(self, channel_idx: int) -> None:
        """
        Open a new, zoomed-in figure for the selected channel.

        Parameters
        ----------
        channel_idx : int
            Which channel to display in the larger view.
        """
        fig, ax = plt.subplots(figsize=(10, 5))
        num_samples = len(self.data[channel_idx, :])
        x_vals = np.linspace(0, self.dt * num_samples, num_samples)
        ax.plot(x_vals, self.data[channel_idx, :])
        ax.set_xlabel("Time (s)")
        ch_idx = channel_idx + self.start_ch
        ax.set_title(f"Channel number: {ch_idx}")


class DistanceTimePlot:
    """
    Creates and manages a distance-time visualization of the signals
    in a single subplot, plus an overlay marking "bad" channels.

    Parameters
    ----------
    data : np.ndarray
        2D NumPy array of shape (n_channels, n_samples).
    dt : float
        Time step (delta) between samples.
    start_ch_index : int
        The index of the first channel for labeling.
    bad_channels : np.ndarray
        Boolean array indicating which channels are currently "bad".
    transp : float, optional
        The alpha (transparency) value for the overlay covering "bad" channels.
    """

    def __init__(
        self,
        data: np.ndarray,
        dt: float,
        start_ch_index: int,
        bad_channels: np.ndarray,
        transp: float = 0.5,
    ) -> None:
        self.data = data
        self.dt = dt
        self.bad_channels = bad_channels
        self.start_ch = start_ch_index
        self.transparency = transp

        # Create main figure and axis
        self.fig, self.ax = plt.subplots()

        # Display the data and add buttons for user interaction
        self._plot_dist_time_data()
        self._add_buttons()

    def update_data(self, data: np.ndarray, start_ch_index: int, bad_channels: np.ndarray) -> None:
        """
        Update the distance-time plot with new data without closing the existing figure.

        Parameters
        ----------
        data : np.ndarray
            New 2D data array for the plot.
        start_ch_index : int
            Index of the first channel for labeling.
        bad_channels : np.ndarray
            Boolean array of bad-channel flags.
        """
        self.data = data
        self.bad_channels = bad_channels
        self.start_ch = start_ch_index
        self.ax.clear()
        self._plot_dist_time_data()

    def _add_buttons(self) -> None:
        """
        Add interactive buttons to the plot area for saving/exiting
        and toggling channel selections.
        """
        self.button_exit = self.fig.add_axes([0.76, 0.005, 0.15, 0.055])
        Button(self.button_exit, "Save & Exit")

        self.button_next = self.fig.add_axes([0.58, 0.005, 0.15, 0.055])
        Button(self.button_next, "Save & Next")

        self.button_selectall = self.fig.add_axes([0.18, 0.005, 0.15, 0.055])
        Button(self.button_selectall, "Toggle All")

    def _plot_dist_time_data(self) -> None:
        """
        Render the distance-time plot. Also create an RGBA overlay
        where alpha in each row is set according to a channel's "bad" status.
        """
        data, start_ch = self.data, self.start_ch
        cmin, cmax = -5, 5
        nd, nt = data.shape

        # Main data plot
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

        # Prepare an overlay array for "bad" channels (RGBA format)
        self.data_overlay = np.zeros((data.shape[0], data.shape[1], 4))
        self.data_overlay[..., 0] = 0.1  # Red component
        self.data_overlay[..., 1] = 0.1  # Green component
        self.data_overlay[..., 2] = 0.1  # Blue component

        # Overlay on top of the data
        self.overlay = self.ax.imshow(
            self.data_overlay,
            extent=[0, nt * self.dt, nd + start_ch, start_ch],
            aspect="auto",
            interpolation="none",
        )
        self.update_overlay(self.bad_channels)

    def update_overlay(self, bad_channels: np.ndarray) -> None:
        """
        Redraw the overlay to highlight "bad" channels.

        Parameters
        ----------
        bad_channels : np.ndarray
            Boolean array indicating which channels to highlight.
        """
        self.bad_channels = bad_channels
        # Fill the alpha channel for each row based on bad-channel flags
        self.data_overlay[..., 3] = np.tile(
            bad_channels[:, np.newaxis] * self.transparency, (1, self.data.shape[1])
        )
        self.overlay.set_data(self.data_overlay)
        self.fig.canvas.draw()


class SelectionPlotter:
    """
    Manages the display of two figures:
    1. A distance-time plot with an overlay for "bad" channels.
    2. A matrix of channel signal subplots.

    Allows interactive clicks to mark channels as "good" or "bad".
    """

    def __init__(
        self,
        data: np.ndarray,
        dt: float = 1,
        start_ch_index: int = 0,
        bad_channels: np.ndarray = None,
    ) -> None:
        self.data = data
        self.dt = dt
        self.bad_channels = bad_channels
        self.start_ch = start_ch_index

        # Initialize bad_channels if not provided
        if self.bad_channels is None:
            self.bad_channels = np.zeros(data.shape[0], dtype=bool)

        # Create both plots
        self.dist_time_plt = DistanceTimePlot(data, dt, self.start_ch, self.bad_channels)
        self.matrix_plt = SignalMatrixPlot(data, dt, self.start_ch, self.bad_channels)

        # Setup event handling for user interactions
        self._enable_onclick_events()

        plt.ion()
        plt.show(block=False)

        # Flags to track the state of the UI
        self.closed = False
        self.single_click_event = None

    def update_data(self, data: np.ndarray, start_ch_index: int, bad_channels: np.ndarray) -> None:
        """
        Update both the distance-time plot and the signal matrix with new data.

        Parameters
        ----------
        data : np.ndarray
            Updated data array of shape (n_channels, n_samples).
        start_ch_index : int
            Updated index for the first channel's label.
        bad_channels : np.ndarray
            Updated boolean flags for bad channels.
        """
        self.data = data
        self.bad_channels = bad_channels
        self.start_ch = start_ch_index

        self.dist_time_plt.update_data(data, start_ch_index, bad_channels)
        self.matrix_plt.update_data(data, start_ch_index, bad_channels)

    def paint_channel(self, channel_idx: int, quality: str) -> None:
        """
        Mark a channel as "bad" or "good" and refresh the plots.

        Parameters
        ----------
        channel_idx : int
            Which channel to mark.
        quality : str
            'good' or 'bad', used for color coding and overlay.
        """
        assert quality in ["good", "bad"]
        changed = False

        # Flip the bad_channels flag if necessary
        if quality == "good" and self.bad_channels[channel_idx]:
            self.bad_channels[channel_idx] = False
            changed = True
        elif quality == "bad" and not self.bad_channels[channel_idx]:
            self.bad_channels[channel_idx] = True
            changed = True

        if not changed:
            return

        # Update visuals
        self.matrix_plt.paint_signal(channel_idx, quality)
        self.dist_time_plt.update_overlay(self.bad_channels)

    def _on_click_in_dist_time_plot(self, event) -> None:
        """
        Handle mouse clicks in the distance-time plot or button areas.

        Left-click marks a channel as "bad", right-click as "good".
        Buttons also trigger events for exit, next, or select-all.
        """
        # If click is inside the main distance-time axes
        if event.inaxes is self.dist_time_plt.ax:
            row = int(event.ydata)
            channel = row - self.start_ch

            if event.button == 1:  # Left mouse
                click_type = "bad"
            elif event.button == 3:  # Right mouse
                click_type = "good"
            else:
                return

            self.paint_channel(channel, click_type)

        # Button: "Save & Exit"
        elif event.inaxes is self.dist_time_plt.button_exit:
            self._on_click_button_close(event)

        # Button: "Save & Next"
        elif event.inaxes is self.dist_time_plt.button_next:
            self._on_click_button_next(event)

        # Button: "Toggle All"
        elif event.inaxes is self.dist_time_plt.button_selectall:
            self._on_click_button_select_all(event)

    def _on_click_in_plot_matrix(self, event) -> None:
        """
        Handle mouse clicks in the signal matrix subplots.

        - Double-click: open a zoomed-in plot for that channel.
        - Single-click: mark channel as "bad" (left) or "good" (right).
        """
        for channel_idx in range(len(self.data)):
            # Check if the click occurred in the axes for this channel
            if self.matrix_plt.axes.flat[channel_idx] == event.inaxes:
                if event.dblclick:
                    # Cancel any pending single-click timer if it exists
                    if self.single_click_event is not None:
                        self.single_click_event.cancel()
                        self.single_click_event = None

                    # Open the zoomed-in plot
                    self.matrix_plt.create_zoomed_plot(channel_idx)
                    return
                else:
                    # Wait for any ongoing single-click event to finish
                    if self.single_click_event is not None:
                        while self.single_click_event.is_alive():
                            time.sleep(0.1)
                        self.single_click_event = None

                    # Determine click type for single-click
                    click_type = "bad" if event.button == 1 else "good"

                    # Use a short delay to detect double-click vs. single-click
                    self.single_click_event = threading.Timer(
                        0.15, self.paint_channel, [channel_idx, click_type]
                    )
                    self.single_click_event.start()
                    return

    def _on_click_button_next(self, _) -> None:
        """Stop the event loop when 'Save & Next' is clicked."""
        self.dist_time_plt.fig.canvas.stop_event_loop()

    def _on_click_button_close(self, _) -> None:
        """Close plots and stop event loop when 'Save & Exit' is clicked."""
        self._on_close_event()
        self.dist_time_plt.fig.canvas.stop_event_loop()

    def _on_click_button_select_all(self, _) -> None:
        """
        Toggle all channels between good and bad.
        If all are bad, switch them all to good; otherwise, switch them all to bad.
        """
        bad_channels = np.ones(self.bad_channels.shape, dtype=bool)
        if self.bad_channels.all():
            bad_channels = np.zeros(self.bad_channels.shape, dtype=bool)

        self.update_data(self.data, self.start_ch, bad_channels)

    def _on_close_event(self, _=None) -> None:
        """Close all figures and mark the plotter as closed."""
        plt.close(self.dist_time_plt.fig)
        plt.close(self.matrix_plt.fig)
        self.closed = True

    def _enable_onclick_events(self) -> None:
        """
        Connect the user-interface events (mouse clicks on plots/buttons,
        figure closing, etc.) to the appropriate handlers.
        """
        # Distance-time figure events
        self.dist_time_plt.fig.canvas.mpl_connect(
            "button_press_event", self._on_click_in_dist_time_plot
        )
        self.dist_time_plt.fig.canvas.mpl_connect("close_event", self._on_close_event)

        # Signal matrix figure events
        self.matrix_plt.fig.canvas.mpl_connect("button_press_event", self._on_click_in_plot_matrix)
        self.matrix_plt.fig.canvas.mpl_connect("close_event", self._on_close_event)

    def get_selections(self) -> np.ndarray:
        """
        Retrieve the latest boolean bad-channel flags as integers (0 or 1).

        Returns
        -------
        np.ndarray
            Array of 0s (good) and 1s (bad) of shape (n_channels,).
        """
        return self.bad_channels.astype(int)

    def start(self) -> None:
        """Start the matplotlib event loop for the distance-time figure."""
        self.dist_time_plt.fig.canvas.start_event_loop(timeout=-1)

    def close(self) -> None:
        """Close figures and mark this plotter as closed."""
        self._on_close_event()


class ChannelSelector:
    """
    Channel Quality Selector that manages data slicing, manual channel selection,
    and saving results. It uses a SelectionPlotter for visualization, along with
    optional SNR-based auto-selection.

    Parameters
    ----------
    data : np.ndarray
        2D array of shape (n_channels, n_samples), i.e., [distance, time].
    dt : float, optional
        Time step (delta) between samples. Default is 1.
    start_channel : int, optional
        Index assigned to the first channel in `data`. Default is 0.
    output_fname : str, optional
        Filename for saving channel quality selections (as CSV). Default is None.

    Examples
    --------
    >>> import numpy as np
    >>> data = np.random.randn(100, 1000)  # 100 channels, 1000 samples
    >>> selector = ChannelSelector(
    ...     data, dt=0.02, start_channel=5, output_fname='out.csv'
    ... )
    >>> selector.auto_select_using_snr(threshold=4)
    >>> selector.select(slice_size=20)
    >>> selections = selector.get_selections()
    >>> selector.save_selections()  # writes to 'out.csv'
    """

    def __init__(
        self, data: np.ndarray, dt: float = 1, start_channel: int = 0, output_fname: str = None
    ) -> None:
        self.data = data
        self.dt = dt
        self.oname = output_fname
        self.start_channel = start_channel

        # By default, no channels are bad (0 means good)
        self.selections = np.zeros(len(data), dtype=int)

    @classmethod
    def read_from_csv(cls, input_filename: str, dt: float = 1) -> "ChannelSelector":
        """
        Initialize a ChannelSelector from data stored in a CSV file.

        Parameters
        ----------
        input_filename : str
            Path to the CSV file containing channels as rows.
        dt : float, optional
            Time step between samples. Default is 1.

        Returns
        -------
        ChannelSelector
            A new instance of ChannelSelector with loaded data.
        """
        data = pd.read_csv(input_filename, sep=",", index_col=0)
        return cls(data.to_numpy(), dt)

    def auto_select_using_snr(self, threshold: float = 5) -> None:
        """
        Automatically mark channels as 'bad' if their crest-factor is
        below a given threshold.
        The crest-factor is defined as max(abs(signal)) / rms(signal).

        Parameters
        ----------
        threshold : float, optional
            Crest-factor threshold for labeling channels as bad. Default is 5.
        """

        def rms(values: np.ndarray) -> float:
            return np.sqrt(np.mean(values**2))

        df_features = pd.DataFrame()
        df_features["rms"] = np.apply_along_axis(rms, 1, self.data)
        df_features["peak"] = np.abs(self.data).max(axis=1)
        df_features["crest-factor"] = df_features["peak"] / df_features["rms"]

        # Mark as 'bad' (1) if crest-factor < threshold, otherwise 'good' (0)
        bad_channels = np.where(df_features["crest-factor"] < threshold, 1, 0)
        self.selections = bad_channels

    def select(self, slice_size: int = 25, channel_shift: int = 0, last_ch: int = None) -> None:
        """
        Interactively mark channels as good/bad in slices of the data.
        Each slice is shown in a SelectionPlotter, allowing manual inspection.

        Parameters
        ----------
        slice_size : int, optional
            Number of channels displayed at once. Default is 25.
        channel_shift : int, optional
            Offset added to channel indices for labeling. Default is 0.
        last_ch : int, optional
            Last channel index to consider. Default is all channels.
        """
        if last_ch is None:
            last_ch = len(self.selections)
        assert last_ch <= len(self.selections), "Last channel is larger than available"
        assert channel_shift < last_ch, "First channel must be smaller than last channel"

        # The first channel in the displayed slice is offset by self.start_channel
        first_ch = channel_shift + self.start_channel

        # Initialize the plotter for the first slice
        plotter = SelectionPlotter(self.data[:slice_size, :], self.dt, first_ch)

        # Iterate over the data in blocks (slices)
        for idx in range(0, self.data.shape[0], slice_size):
            # If the user closes the plotter, stop immediately
            if plotter.closed:
                print("Channel selector closed at index:", idx - 1)
                break

            # Get the current block of data and channel indices
            data_slice = self.data[idx : idx + slice_size, :]
            start_ch = idx + first_ch
            bad_channels_slice = self.selections[idx : idx + slice_size]

            # Update plotter visuals and wait for user interactions
            plotter.update_data(data_slice, start_ch, bad_channels_slice)
            plotter.start()

            # After the user closes the plotter's event loop,
            # retrieve the final selections
            manual_selections = plotter.get_selections()
            self.selections[idx : idx + slice_size] = manual_selections

            # Save intermediate results to file
            self.save_selections()

        # Close the plotter once everything is done
        plotter.close()

    def get_selections(self, first_ch: int = None, last_ch: int = None) -> pd.DataFrame:
        """
        Retrieve a DataFrame of the channel selections.

        Parameters
        ----------
        first_ch : int, optional
            First channel index to return. Defaults to self.start_channel.
        last_ch : int, optional
            Last channel index to return. Defaults to the maximum channel index.

        Returns
        -------
        pd.DataFrame
            A DataFrame with one column, 'Quality', containing 0 for good and 1 for bad.
        """
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

        index = range(first_ch, last_ch)
        data = self.selections[first_ch - self.start_channel : last_ch - self.start_channel]
        df = pd.DataFrame(data, index=index, columns=["Quality"])
        return df

    def save_selections(self, output_fname: str = None) -> None:
        """
        Save the current selections to a CSV file.

        Parameters
        ----------
        output_fname : str, optional
            The path to the output file; if None, uses the constructor's output_fname.
        """
        if self.oname is None and output_fname is None:
            return
        if output_fname is None:
            output_fname = self.oname

        selections_df = self.get_selections()
        selections_df.to_csv(output_fname, sep=",")


if __name__ == "__main__":
    filename = "data/SAFE/ICM_2023-11-13_14-57-38.h5"

    # Import raw data from an .h5 file
    data = utl.import_h5(filename)

    # Example filtering and decimation
    freq = 100
    data = utl.filter_imported_data(data, pass_low=3, pass_hi=12.5, freq=freq, decimate_data=True)
    start_channel = 236
    data = utl.set_data_limits(data, first_ch=start_channel, first_time=3000, last_time=6500)
    data = data.T.to_numpy()

    # Instantiate the selector
    selector = ChannelSelector(data, 1.0 / 50, start_channel, output_fname="quality_mark.csv")

    # Automatic selection using a crest-factor threshold
    selector.auto_select_using_snr(threshold=3)

    # Interactive channel-by-channel inspection
    selector.select(slice_size=36, channel_shift=0)

    # Print final selections to console
    print(selector.get_selections())

    # Save final selections with a timestamped filename
    timestamp = datetime.now().strftime("%m%d_%H%M")
    selector.save_selections(
        "channel_selections/CQI_selections_"
        + filename.split("/")[-1].split(".")[1]
        + "_picked_"
        + timestamp
        + ".csv"
    )
