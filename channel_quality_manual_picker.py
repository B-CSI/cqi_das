import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import h5py
import utils as utl
from pathlib import Path
from datetime import datetime


def plot_per_channel_matrix(data, num_channels=None, ch0=0, start_ch=0, bad_channels=None):
    """Create a square matrix plot with per-channel signals."""
    if num_channels is None:
        num_channels = data.shape[0]
    x = int(np.sqrt(num_channels))  # make it a 10 by 10 visual matrix
    sz = 10  # matplotlib size of figure
    fig, ax = plt.subplots(nrows=x, ncols=x, figsize=(sz + sz / 2, sz))
    print('start_ch=',start_ch)

    for i, row in enumerate(ax):
        for j, col in enumerate(row):
            k = i * x + j
            if k >= num_channels:
                continue

            if bad_channels is not None and bad_channels[k]:
                color = "C1"
            else:
                color = "C0"
            col.plot(data[ch0 + k], color=color)
            col.title.set_text(str(start_ch + k))

    fig.tight_layout()
    return fig, ax


def find_bad_channels(data):
    """Find bad channels automatically according to a criterion."""

    def rms(values):
        return np.sqrt(np.mean(np.square(values)))

    df_features = pd.DataFrame()
    df_features["rms"] = np.apply_along_axis(rms, 1, data)
    df_features["peak"] = np.abs(data).max(axis=1)
    df_features["crest-factor"] = df_features["peak"] / df_features["rms"]

    threshold = 5  # Manually set threshold
    bad_channels = np.where(df_features["crest-factor"] < threshold, 0, 0)
    return bad_channels


def channel_picker(data, dt=None, start_ch=0):
    """
    Opens a matplotlib plot for interactive channel selection,
    allowing the user to label each channel as "good" or "bad".

    - Channels are good by default.
    - Use left click to label as "bad".
    - Use right click to label back as "good".

    Parameters:
    -----------
    data : numpy.ndarray
        A 2D numpy array where the first axis represents
        the channels and the second axis represents time.
    dt : float, optional
        The time step or time unit for the time axis. If provided,
        it will be used to scale the x-axis of the plot.
        Default is None.

    Returns:
    --------
    numpy.ndarray
        A 1D numpy array where each element corresponds to a channel
        in the input data, labeled as 1 for "bad" and 0 for "good".
    """

    if dt is None:
        dt = 1

    cmin, cmax = -5, 5
    nd, nt = data.shape
    display_data = np.copy(data)

    # Pre-pick some bad channels automatically
    prepicked_bad_channels = find_bad_channels(data)

    # Dictionary to store clicks
    clicks = {}
    for ch in range(prepicked_bad_channels.shape[0]):
        if prepicked_bad_channels[ch]:
            clicks[ch] = "bad"
            display_data[ch] = cmin

    fig, ax = plt.subplots()

    # Create per-channel plot matrix
    fig_matrix, ax_matrix = plot_per_channel_matrix(
        data, num_channels=nd, start_ch=start_ch, bad_channels=prepicked_bad_channels
    )
    matrix_side = int(np.sqrt(nd))

    def draw_images():
        ax.imshow(
            display_data,
            aspect="auto",
            cmap="seismic",
            extent=[0, nt * dt, nd + start_ch, start_ch],
            vmin=cmin,
            vmax=cmax,
            interpolation="none",
        )
        fig.canvas.draw()
        fig_matrix.canvas.draw()

    # Function to handle mouse click events
    def on_click_in_img(event):
        """Handle clicks in the channel vs time plot."""
        if event.inaxes:
            row = int(event.ydata)
            col = int(event.xdata)
            matrix_plot_ax = ax_matrix[row // matrix_side, row % matrix_side]

            # remove existing lines
            for line in matrix_plot_ax.get_lines():
                line.remove()

            if event.button == 1:  # Left click
                click_type = "bad"
                display_data[row, :] = cmin
                matrix_plot_ax.plot(data[row, :], color="C1")
            elif event.button == 3:  # Right click
                click_type = "good"
                display_data[row, :] = data[row, :]
                matrix_plot_ax.plot(data[row, :], color="C0")
            else:
                return  # Ignore other clicks

            # Store the click in the dictionary
            clicks[row] = click_type
            print(f"Row: {row}, Click: {click_type}")

            draw_images()

    def on_click_in_matrix(event):
        """Handle clicks in the per-channel signal matrix plot."""
        found_row = None
        # Find the axes that was clicked on
        for i, row in enumerate(ax_matrix):
            for j, col in enumerate(row):
                k = i * matrix_side + j
                if col == event.inaxes:
                    found_row = k
                    if event.button == 1:  # Left click
                        click_type = "bad"
                        display_data[k, :] = cmin
                    elif event.button == 3:  # Right click
                        click_type = "good"
                        display_data[k, :] = data[k, :]

                    lines = col.get_lines()
                    if lines:
                        # Change color of the plot
                        color = "C1" if event.button == 1 else "C0"
                        lines[0].set_color(color)
                    break

        if found_row is None:
            return

        # Store the click in the dictionary
        clicks[found_row] = click_type
        #print(f"Row: {found_row}, Click: {click_type}")

        draw_images()

    draw_images()
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Channel number")

    # Connect the click event to the handler function
    fig.canvas.mpl_connect("button_press_event", on_click_in_img)
    fig_matrix.canvas.mpl_connect("button_press_event", on_click_in_matrix)

    # Show the plot
    plt.show()

    # Set the labels, 0 = good channel
    labels = np.zeros(nd)
    for row_num in range(nd):
        if row_num in clicks and clicks[row_num] == "bad":
            labels[row_num] = 1  # 1 = bad channel

    return labels


if __name__ == "__main__":
    filepath = Path('/home/tatiana/Documents/Onboarding_to_TREMORS/CANDAS/GC/20200815_19h05m/eventCANDAS/').absolute()
    data = utl.import_miniseed(filepath, 2020, 'G')
    first_ch = 415
    last_ch = data.shape[1]
    first_time = 13000
    last_time = 18000
    data = utl.set_data_limits(data, first_ch=first_ch, last_ch=last_ch, first_time=first_time, last_time=last_time)
    data = utl.filter_imported_data(data, decimate_data=True)
    data = data.T.to_numpy()
    dt = 0.04

    # Iterate channel slices
    slice_size = 25
    qualities = []
    for idx in range(0, data.shape[0], slice_size):
        data_slice = data[idx : idx + slice_size, :]
        start_ch = idx + first_ch
        print('Channel', str(start_ch))
        picks = channel_picker(data_slice, dt, start_ch)
        print(picks)
        qualities.append(picks)

    quality_labels = np.concatenate(qualities).astype(int)
    print("Final picks:")
    print(quality_labels)
    timestamp = datetime.now().strftime("%m%d_%H%M")

    np.savetxt("results/quality_labels" + "_" + timestamp + ".txt", quality_labels, delimiter=",", fmt="%d")
