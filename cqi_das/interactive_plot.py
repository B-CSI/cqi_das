"""
Copyright (c) 2025, Spanish National Research Council (CSIC)
Developed by the Barcelona Center for Subsurface Imaging (B-CSI)

This source code is subject to the terms of the
GNU Lesser General Public License.
"""

import matplotlib.pyplot as plt
from matplotlib.backend_bases import MouseEvent
import numpy as np
import pandas as pd
import threading
import time
import os


class DraggableThreshold:
    """
    Handles a draggable vertical threshold line in a specified matplotlib Axes.

    The figure is updated only after the mouse is released.
    """

    def __init__(
        self,
        ax_mid: plt.Axes,
        line: plt.Line2D,
        update_callback: callable,
        step: float = 0.01,
    ) -> None:
        """
        Initialize a DraggableThreshold object.

        Parameters
        ----------
        ax_mid : plt.Axes
            Axes where the threshold line is drawn.
        line : plt.Line2D
            The vertical line object (axvline) representing the threshold.
        update_callback : callable
            Callback called on mouse release with the new threshold value.
        step : float, optional
            Step size for snapping the threshold (default is 0.01).
        """
        self.ax_mid = ax_mid
        self.line = line
        self.update_callback = update_callback
        self.step = step

        self.press = None  # Tracks the x-position when clicked
        self.threshold_init = line.get_xdata()[0]
        self.current_threshold = self.threshold_init

        # Connect mouse event callbacks
        self.cid_press = line.figure.canvas.mpl_connect("button_press_event", self.on_press)
        self.cid_release = line.figure.canvas.mpl_connect("button_release_event", self.on_release)
        self.cid_motion = line.figure.canvas.mpl_connect("motion_notify_event", self.on_motion)

        self._resize_timer = None
        self._resize_last_time = 0
        self.cid_resize = line.figure.canvas.mpl_connect("resize_event", self.on_resize)
        self.update_background_ax_mid()

    def on_resize(self, event):
        """When resizing ends, cache the background"""
        if self._resize_timer is not None:
            self._resize_timer.cancel()

        # Start a new timer
        self._resize_last_time = time.time()
        self._resize_timer = threading.Timer(0.5, self._debounced_update_background)
        self._resize_timer.start()

    def _debounced_update_background(self):
        """
        Runs after 0.5s of no new resize_event.
        """
        # We can do final checks, or just recache directly:
        self.update_background_ax_mid()

    def update_background_ax_mid(self):
        """
        Update and cache the background of ax_mid. This is important for smooth
        interactions.
        """

        self.line.set_visible(False)
        self.line.figure.canvas.draw()

        self.background_mid = self.line.figure.canvas.copy_from_bbox(self.ax_mid.bbox)

        self.line.set_visible(True)
        self.line.figure.canvas.draw()

    def on_press(self, event: MouseEvent) -> None:
        """
        Record initial reference if click is near the threshold line.

        Parameters
        ----------
        event : plt.MouseEvent
            The mouse press event.
        """
        if event.inaxes != self.ax_mid:
            return

        x0 = self.line.get_xdata()[0]
        if abs(event.xdata - x0) < 0.2:  # Tolerance in data coordinates
            self.press = (x0, event.x)

    def on_motion(self, event: MouseEvent) -> None:
        """
        Drag the threshold line with the mouse (lightweight update).

        Parameters
        ----------
        event : plt.MouseEvent
            The mouse movement event.
        """
        if self.press is None or event.inaxes != self.ax_mid:
            return

        x0, xpress = self.press
        dx = event.xdata - x0
        new_x = x0 + dx

        # Snap to specified step and clamp to [0, 1]
        new_x = round(new_x / self.step) * self.step
        new_x = max(0.0, min(1.0, new_x))

        if abs(new_x - self.current_threshold) < 0.01:
            # Skip if the line hasn’t moved enough
            return

        # Before drawing, restore the cached background
        if self.background_mid is not None:
            self.line.figure.canvas.restore_region(self.background_mid)

        # Update line position
        self.line.set_xdata([new_x, new_x])

        # Draw only the line artist
        self.ax_mid.draw_artist(self.line)

        # Blit the region
        self.line.figure.canvas.blit(self.ax_mid.bbox)

        # self.line.figure.canvas.draw_idle()

    def on_release(self, event: MouseEvent) -> None:
        """
        Finalize threshold update when mouse is released.

        Parameters
        ----------
        event : plt.MouseEvent
            The mouse release event.
        """
        if self.press is None:
            return

        final_threshold = self.line.get_xdata()[0]
        self.press = None
        self.current_threshold = final_threshold
        self.update_callback(final_threshold)

        # self.line.figure.canvas.draw()
        self.update_background_ax_mid()


def update_event_selection_plots(
    ax_left: plt.Axes,
    ax_right: plt.Axes,
    ax_mid: plt.Axes,
    data_df: pd.DataFrame,
    channel_probs: pd.Series,
    channel_nums: pd.Index,
    im_left: plt.Axes.imshow,
    im_right: plt.Axes.imshow,
    threshold: float,
) -> None:
    """
    Update "bad" and "good" channel plots based on a new threshold.

    Parameters
    ----------
    ax_left : plt.Axes
        Axes displaying low-quality (bad) channels.
    ax_right : plt.Axes
        Axes displaying high-quality (good) channels.
    ax_mid : plt.Axes
        Axes displaying channel probabilities.
    data_df : pd.DataFrame
        Original data with rows as time and columns as channels.
    channel_probs : pd.Series
        Probabilities indicating high quality for each channel.
    channel_nums : pd.Index
        Channel labels or numbers.
    im_left : matplotlib.image.AxesImage
        The image object for low-quality channel visualization.
    im_right : matplotlib.image.AxesImage
        The image object for high-quality channel visualization.
    threshold : float
        Threshold value to separate low and high quality channels.

    Returns
    -------
    None
    """
    bad_ch_data = data_df.copy()
    good_ch_data = data_df.copy()

    cols_to_zero_bad = channel_nums[channel_probs > threshold]
    bad_ch_data.loc[:, cols_to_zero_bad] = 0

    cols_to_zero_good = channel_nums[channel_probs < threshold]
    good_ch_data.loc[:, cols_to_zero_good] = 0

    im_left.set_data(bad_ch_data.T)
    im_right.set_data(good_ch_data.T)


def create_interactive_plot(
    data_df: pd.DataFrame,
    channel_probs: pd.Series,
    channel_nums: pd.Index,
    initial_threshold: float = 0.6,
    show_indications: bool = True,
    save_path: str = None,
    figure_size: tuple[int, int] = (10, 10),
    vmin: float = -5,
    vmax: float = 5,
) -> tuple[plt.Figure, DraggableThreshold]:
    """
    Create an interactive figure for visualizing channel-quality probabilities.

    A draggable vertical line in the middle plot allows the user to adjust
    the threshold, automatically updating "bad" and "good" channel plots.

    Parameters
    ----------
    data_df : pd.DataFrame
        Data with rows as time samples and columns as channels.
    channel_probs : pd.Series
        Probabilities for each channel (e.g. quality metric).
    channel_nums : pd.Index
        Index or labels for each channel.
    initial_threshold : float, optional
        Starting value for the draggable threshold (default is 0.6).

    Returns
    -------
    fig : plt.Figure
        The created matplotlib Figure.
    dragger : DraggableThreshold
        An instance of DraggableThreshold to handle interactive dragging.
    """
    fig, (ax_left, ax_mid, ax_right) = plt.subplots(
        1,
        3,
        figsize=figure_size,
        gridspec_kw={"width_ratios": [8, 2, 8]},
        sharey=True,
    )

    dummy_data = np.zeros_like(data_df)

    # Derive time axis extent for demonstration
    first_sec, last_sec = 0, int(dummy_data.shape[0] / 50)
    first_ch, last_ch = channel_nums.min(), channel_nums.max()

    # Left: "Bad" channels placeholder
    im_left = ax_left.imshow(
        dummy_data.T,
        vmin=vmin,
        vmax=vmax,
        aspect="auto",
        cmap="seismic",
        interpolation="none",
        extent=[first_sec, last_sec, last_ch, first_ch],
    )
    ax_left.set_title("Low Quality Channels")
    ax_left.set_xlabel("Time (s)")
    ax_left.set_ylabel("Channel number")

    # Right: "Good" channels placeholder
    im_right = ax_right.imshow(
        dummy_data.T,
        vmin=vmin,
        vmax=vmax,
        aspect="auto",
        cmap="seismic",
        interpolation="none",
        extent=[first_sec, last_sec, last_ch, first_ch],
    )
    ax_right.set_title("High Quality Channels")
    ax_right.set_xlabel("Time (s)")

    # Middle: channel probabilities + threshold line
    ax_mid.scatter(channel_probs, channel_nums, s=3, alpha=0.7, color="grey")
    ax_mid.set_xlim([0, 1])
    ax_mid.set_title("Channel Probability\n(Low vs. High)")
    ax_mid.set_xlabel("Probability (High)")

    threshold_line = ax_mid.axvline(initial_threshold, color="maroon", linestyle="--", linewidth=2)
    if show_indications:
        text_threshold_info = fig.suptitle(
            f"Current threshold: {initial_threshold:.2f}. Drag the line to change! Threshold is saved automatically."
        )
    else:
        text_threshold_info = None

    def on_threshold_release(new_threshold: float) -> None:
        update_event_selection_plots(
            ax_left,
            ax_right,
            ax_mid,
            data_df,
            channel_probs,
            channel_nums,
            im_left,
            im_right,
            new_threshold,
        )
        if text_threshold_info is not None:
            text_threshold_info.set_text(
                f"Current threshold: {new_threshold:.2f}. Drag the line to change! Threshold is saved automatically."
            )

    # Attach the draggable threshold
    dragger = DraggableThreshold(ax_mid, threshold_line, on_threshold_release, step=0.01)

    # Initialize the plots with the provided threshold
    on_threshold_release(initial_threshold)

    fig.tight_layout()
    dragger.update_background_ax_mid()

    if not show_indications and save_path is not None:
        absolute_path = os.path.abspath(save_path)
        print(f"Saving plot to {absolute_path}")
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig, dragger
