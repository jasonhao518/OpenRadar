#!/usr/bin/env python3
"""Visualize TI UART range profiles or point clouds with a frame slider."""

import argparse

import matplotlib.pyplot as plt
from matplotlib.widgets import Button, Slider

from mmwave.dataloader.ti_uart import (
    TI_TLV_DETECTED_POINTS,
    TI_TLV_RANGE_PROFILE_MAJOR,
    load_range_profiles,
    load_ti_uart_frames,
)


def controls(figure, count, update):
    slider_axis = figure.add_axes([0.20, 0.06, 0.60, 0.04])
    slider = Slider(slider_axis, "Frame", 0, count - 1, valinit=0, valstep=1)
    previous = Button(figure.add_axes([0.05, 0.05, 0.10, 0.06]), "Previous")
    next_button = Button(figure.add_axes([0.85, 0.05, 0.10, 0.06]), "Next")
    slider.on_changed(lambda value: update(int(value)))
    previous.on_clicked(lambda _event: slider.set_val(max(0, int(slider.val) - 1)))
    next_button.on_clicked(
        lambda _event: slider.set_val(min(count - 1, int(slider.val) + 1))
    )
    return slider, previous, next_button


def show_range_profiles(capture):
    frame_numbers, profiles = load_range_profiles(capture)
    figure, axis = plt.subplots()
    figure.subplots_adjust(bottom=0.18)
    (line,) = axis.plot(profiles[0], linewidth=1.5)
    axis.set(xlabel="Range bin", ylabel="Magnitude", xlim=(0, profiles.shape[1] - 1))
    axis.set_ylim(0, max(1, profiles.max()) * 1.05)
    axis.grid(alpha=0.25)
    title = axis.set_title("Frame {} (1/{})".format(frame_numbers[0], len(profiles)))

    def update(index):
        line.set_ydata(profiles[index])
        title.set_text("Frame {} ({}/{})".format(frame_numbers[index], index + 1, len(profiles)))
        figure.canvas.draw_idle()

    widgets = controls(figure, len(profiles), update)
    plt.show()
    return widgets


def show_point_cloud(capture):
    frames = [
        frame for frame in load_ti_uart_frames(capture)
        if frame.payloads(TI_TLV_DETECTED_POINTS)
    ]
    if not frames:
        raise ValueError("no compressed point-cloud TLV 301 found")
    clouds = [frame.point_cloud() for frame in frames]
    limit = max(
        1.0,
        max((abs(value) for cloud in clouds for axis in ("x", "y") for value in cloud[axis]), default=1.0),
    ) * 1.1
    figure, axis = plt.subplots()
    figure.subplots_adjust(bottom=0.18)
    scatter = axis.scatter(clouds[0]["x"], clouds[0]["y"], c=clouds[0]["snr"], s=30)
    axis.set(xlabel="x (m)", ylabel="y (m)", xlim=(-limit, limit), ylim=(-limit, limit))
    axis.set_aspect("equal", adjustable="box")
    axis.grid(alpha=0.25)
    title = axis.set_title("Frame {} (1/{})".format(frames[0].header.frame_number, len(frames)))

    def update(index):
        cloud = clouds[index]
        scatter.set_offsets(list(zip(cloud["x"], cloud["y"])))
        scatter.set_array(cloud["snr"])
        title.set_text(
            "Frame {} ({}/{})".format(frames[index].header.frame_number, index + 1, len(frames))
        )
        figure.canvas.draw_idle()

    widgets = controls(figure, len(frames), update)
    plt.show()
    return widgets


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture", help="binary TI UART TLV capture")
    parser.add_argument("--view", choices=("range", "points"), default="range")
    args = parser.parse_args()
    if args.view == "points":
        show_point_cloud(args.capture)
    else:
        show_range_profiles(args.capture)


if __name__ == "__main__":
    main()
