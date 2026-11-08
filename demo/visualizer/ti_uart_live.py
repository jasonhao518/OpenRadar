#!/usr/bin/env python3
"""Live TI UART point-cloud, range-profile, and depth-map viewer."""

import argparse
import time
import queue
import threading
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import serial
from matplotlib.animation import FuncAnimation

from mmwave.dataloader.ti_uart import (
    TI_TLV_DETECTED_POINTS,
    TI_TLV_RANGE_AZIMUTH_HEATMAP_MAJOR,
    TI_TLV_RANGE_PROFILE_MAJOR,
    TIUARTStreamParser,
)
from mmwave.dsp.environment_map import ReflectionMap


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", required=True, help="serial device")
    parser.add_argument(
        "--baud", type=int, default=128000,
        help="data baud when --config is not used (default: 128000)",
    )
    parser.add_argument("--cli-baud", type=int, default=115200)
    parser.add_argument(
        "--config", type=Path,
        help="send a TI .cfg at --cli-baud before starting the live view",
    )
    parser.add_argument("--command-timeout", type=float, default=2.0)
    parser.add_argument(
        "--view", choices=("points", "range", "both", "depth", "depth3d", "map", "scene", "all"),
        default="both",
    )
    parser.add_argument("--limit", type=float, default=5.0, help="point-cloud axis limit in metres")
    parser.add_argument(
        "--azimuth-bins", type=int, default=32,
        help="azimuth FFT width in TLV 304 (default: 32)",
    )
    parser.add_argument(
        "--range-resolution", type=float, default=0.1953125,
        help="metres per range bin for depth view (default: 0.1953125)",
    )
    parser.add_argument(
        "--min-range", type=float, default=0.25,
        help="ignore nearer bins in derived depth profile (default: 0.25 m)",
    )
    parser.add_argument(
        "--max-range", type=float, default=10.0,
        help="depth-view range limit (default: 10 m)",
    )
    parser.add_argument("--interval", type=int, default=50, help="UI refresh interval in ms")
    parser.add_argument("--record", type=Path, help="also save received bytes to this file")
    parser.add_argument("--map-history", type=float, default=30.0, help="map evidence time constant in seconds")
    parser.add_argument("--map-threshold", type=float, default=6.0, help="peak contrast above median background in dB")
    parser.add_argument("--fps", type=float, default=4.0, help="configured radar FPS, used for map aging")
    return parser


def read_cli_reply(serial_port, timeout):
    reply = bytearray()
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        waiting = serial_port.in_waiting
        chunk = serial_port.read(min(max(waiting, 1), 4096))
        if chunk:
            reply.extend(chunk)
            if b"mmwDemo:/>" in reply:
                break
        else:
            time.sleep(0.005)
    return bytes(reply)


def send_config(serial_port, config_path, timeout):
    commands = []
    for line in config_path.read_text(encoding="utf-8").splitlines():
        command = line.strip()
        if command and not command.startswith(("%", "#")):
            commands.append(command)

    print("Sending {} commands from {}".format(len(commands), config_path), flush=True)
    for command in commands:
        print("> {}".format(command), flush=True)
        serial_port.write((command + "\n").encode("ascii"))
        serial_port.flush()
        words = command.split()

        if words[0] == "baudRate" and len(words) == 2:
            data_baud = int(words[1])
            time.sleep(0.25)
            serial_port.baudrate = data_baud
            # Synchronize explicitly at the new baud. The command's original
            # echo can straddle the hardware UART close/open transition.
            serial_port.reset_input_buffer()
            serial_port.write(b"\n")
            serial_port.flush()
            reply = read_cli_reply(serial_port, timeout)
            if b"mmwDemo:/>" not in reply:
                raise TimeoutError(
                    "no CLI prompt after switching host UART to {}".format(data_baud)
                )
            print("  host UART switched to {} baud".format(data_baud), flush=True)
        elif words[0] == "sensorStart":
            # CLI echo and binary TLVs may be adjacent. Leave all input for the
            # incremental magic-word parser rather than draining it here.
            time.sleep(0.1)
            return
        else:
            reply = read_cli_reply(serial_port, timeout)

        if b"Error:" in reply or b"Error -" in reply:
            raise RuntimeError(
                "radar rejected {!r}: {}".format(
                    command, reply.decode("utf-8", errors="replace").strip()
                )
            )
        if words[0] != "baudRate" and b"mmwDemo:/>" not in reply:
            raise TimeoutError("no CLI prompt after {!r}".format(command))


class LiveViewer:
    def __init__(self, args):
        self.args = args
        initial_baud = args.cli_baud if args.config else args.baud
        self.serial = serial.Serial(args.port, initial_baud, timeout=0)
        self.serial.reset_input_buffer()
        if args.config:
            send_config(self.serial, args.config, args.command_timeout)
        self.parser = TIUARTStreamParser()
        self.record = args.record.open("wb") if args.record else None
        self.bytes_received = 0
        self.frames_received = 0
        self.last_frame = None
        self.error = None
        self.skipped_payloads = 0
        self.pending = queue.Queue()
        self.stop_reader = threading.Event()
        self.serial.timeout = 0.05
        self.reader = threading.Thread(target=self.receive, daemon=True)
        self.reader.start()

        show_points = args.view in ("points", "both", "all")
        show_range = args.view in ("range", "both", "all")
        show_depth = args.view in ("depth", "depth3d", "scene", "all")
        show_3d = args.view in ("depth3d", "scene", "all")
        show_map = args.view in ("map", "scene", "all")
        panel_count = sum((show_points, show_range, show_depth, show_3d, show_map))
        self.figure = plt.figure(figsize=(6 * panel_count, 6))
        panel = iter(range(1, panel_count + 1))
        self.point_axis = self.figure.add_subplot(1, panel_count, next(panel)) if show_points else None
        self.range_axis = self.figure.add_subplot(1, panel_count, next(panel)) if show_range else None
        self.depth_axis = self.figure.add_subplot(1, panel_count, next(panel)) if show_depth else None
        self.cloud_axis = self.figure.add_subplot(1, panel_count, next(panel), projection="3d") if show_3d else None
        self.map_axis = self.figure.add_subplot(1, panel_count, next(panel)) if show_map else None
        self.environment = None
        self.map_image = None
        if show_map:
            self.environment = ReflectionMap(
                limit=args.limit, max_range=args.max_range,
                range_step=args.range_resolution, fps=args.fps,
                history=args.map_history, threshold_db=args.map_threshold,
            )
            cmap = plt.get_cmap("inferno").copy()
            cmap.set_bad("#343a40")
            self.map_image = self.map_axis.imshow(
                np.ma.masked_all(self.environment.evidence.shape), origin="lower",
                extent=(-args.limit, args.limit, 0, args.max_range),
                cmap=cmap, vmin=0, vmax=1, aspect="equal", interpolation="nearest",
            )
            self.map_axis.plot(0, 0, "c^", markersize=9)
            self.map_axis.set(xlabel="x / lateral (m)", ylabel="y / forward (m)",
                title="Accumulated reflection map\nGray = unknown; R = clear")
            self.figure.colorbar(self.map_image, ax=self.map_axis, pad=0.02).set_label("Reflection persistence (not occupancy probability)")
            self.figure.canvas.mpl_connect("key_press_event", self.map_key)
        self.cloud_scatter = None
        if self.cloud_axis is not None:
            self.cloud_scatter = self.cloud_axis.scatter(
                [], [], [], c=[], s=25, vmin=0, vmax=40, cmap="viridis"
            )
            self.cloud_axis.set(
                xlabel="x / lateral (m)", ylabel="y / forward (m)",
                zlabel="z / height (m)", xlim=(-args.limit, args.limit),
                ylim=(0, args.max_range), zlim=(-args.limit, args.limit),
                title="3D detected point cloud (TLV 301)",
            )
            self.cloud_axis.view_init(elev=20, azim=-65)
            self.figure.colorbar(self.cloud_scatter, ax=self.cloud_axis, pad=0.1).set_label("SNR (dB)")
        self.figure.canvas.manager.set_window_title("OpenRadar TI UART live viewer")
        self.figure.canvas.mpl_connect("close_event", self.close)

        self.scatter = None
        if self.point_axis is not None:
            self.scatter = self.point_axis.scatter([], [], c=[], s=40, vmin=0, vmax=40)
            colorbar = self.figure.colorbar(self.scatter, ax=self.point_axis, pad=0.02)
            colorbar.set_label("SNR (dB)")
            self.point_axis.set(
                xlabel="x (m)", ylabel="y (m)",
                xlim=(-args.limit, args.limit), ylim=(0, args.limit),
                title="Compressed point cloud (TLV 301)",
            )
            self.point_axis.set_aspect("equal", adjustable="box")
            self.point_axis.grid(alpha=0.25)

        self.range_line = None
        if self.range_axis is not None:
            (self.range_line,) = self.range_axis.plot([], [], linewidth=1.5)
            self.range_axis.set(
                xlabel="Range bin", ylabel="Magnitude",
                xlim=(0, 255), ylim=(0, 1),
                title="Major range profile (TLV 302)",
            )
            self.range_axis.grid(alpha=0.25)

        self.depth_image = None
        self.depth_line = None
        if self.depth_axis is not None:
            self.depth_image = self.depth_axis.imshow(
                np.zeros((2, args.azimuth_bins)),
                origin="lower",
                aspect="auto",
                cmap="turbo",
                extent=(-90, 90, 0, args.max_range),
            )
            (self.depth_line,) = self.depth_axis.plot(
                [], [], color="white", linewidth=1.5, label="strongest return"
            )
            self.depth_axis.set(
                xlabel="Azimuth angle (deg)", ylabel="Range / depth (m)",
                xlim=(-90, 90), ylim=(0, args.max_range),
                title="Dense range-azimuth depth view (TLV 304)",
            )
            self.depth_axis.legend(loc="upper right")
            colorbar = self.figure.colorbar(
                self.depth_image, ax=self.depth_axis, pad=0.02
            )
            colorbar.set_label("log(1 + return intensity)")

        self.status = self.figure.suptitle(self._status_text())
        self.figure.tight_layout(rect=(0, 0, 1, 0.94))
        self.animation = FuncAnimation(
            self.figure,
            self.poll,
            interval=args.interval,
            blit=False,
            cache_frame_data=False,
        )

    def _status_text(self):
        if self.error:
            return "Serial error: {}".format(self.error)
        frame = "waiting" if self.last_frame is None else str(self.last_frame)
        return "{} @ {:,} baud  |  frame {}  |  {:,} bytes  |  {:,} packets | skipped {}".format(
            self.args.port,
            self.serial.baudrate,
            frame,
            self.bytes_received,
            self.frames_received,
            self.parser.malformed_packets + self.skipped_payloads,
        )

    def receive(self):
        """Drain the OS serial buffer even while Matplotlib is drawing."""
        try:
            while not self.stop_reader.is_set():
                chunk = self.serial.read(min(max(self.serial.in_waiting, 1), 65536))
                if chunk:
                    self.bytes_received += len(chunk)
                    if self.record:
                        self.record.write(chunk)
                        self.record.flush()
                    self.pending.put(chunk)
        except (OSError, serial.SerialException) as error:
            if not self.stop_reader.is_set():
                self.error = error

    def poll(self, _tick):
        try:
            for _ in range(256):
                try:
                    chunk = self.pending.get_nowait()
                except queue.Empty:
                    break
                for frame in self.parser.feed(chunk):
                    try:
                        self.update_frame(frame)
                    except (ValueError, IndexError):
                        self.skipped_payloads += 1
        except (OSError, serial.SerialException, ValueError) as error:
            self.error = error
        self.status.set_text(self._status_text())
        return self.artists

    @property
    def artists(self):
        return tuple(
            artist for artist in (
                self.scatter, self.range_line, self.depth_image,
                self.depth_line, self.cloud_scatter, self.map_image, self.status,
            )
            if artist is not None
        )

    def update_frame(self, frame):
        self.frames_received += 1
        self.last_frame = frame.header.frame_number

        if self.environment is not None and frame.payloads(TI_TLV_RANGE_AZIMUTH_HEATMAP_MAJOR):
            evidence = self.environment.update(
                frame.range_azimuth_heatmap(self.args.azimuth_bins),
                frame.header.frame_number, self.args.min_range,
            )
            self.map_image.set_data(np.ma.masked_less(evidence, 0.15))

        if self.cloud_scatter is not None:
            if frame.payloads(TI_TLV_DETECTED_POINTS):
                points = frame.point_cloud()
                self.cloud_scatter._offsets3d = (points["x"], points["y"], points["z"])
                self.cloud_scatter.set_array(points["snr"])
            else:
                self.cloud_scatter._offsets3d = ([], [], [])
                self.cloud_scatter.set_array(np.empty(0))

        if self.scatter is not None:
            if frame.payloads(TI_TLV_DETECTED_POINTS):
                points = frame.point_cloud()
                offsets = np.column_stack((points["x"], points["y"]))
                self.scatter.set_offsets(offsets)
                self.scatter.set_array(points["snr"])
            else:
                self.scatter.set_offsets(np.empty((0, 2)))
                self.scatter.set_array(np.empty(0))

        if self.range_line is not None and frame.payloads(TI_TLV_RANGE_PROFILE_MAJOR):
            profile = frame.range_profile()
            self.range_line.set_data(np.arange(len(profile)), profile)
            self.range_axis.set_xlim(0, max(1, len(profile) - 1))
            current_top = self.range_axis.get_ylim()[1]
            target_top = max(1, float(profile.max()) * 1.05)
            if target_top > current_top or target_top < current_top * 0.5:
                self.range_axis.set_ylim(0, target_top)

        if (
            self.depth_image is not None
            and frame.payloads(TI_TLV_RANGE_AZIMUTH_HEATMAP_MAJOR)
        ):
            heatmap = frame.range_azimuth_heatmap(self.args.azimuth_bins)
            # TI emits unshifted FFT bins. Center broadside for a conventional
            # left-to-right -90..+90 degree depth image.
            heatmap = np.fft.fftshift(heatmap, axes=1)
            image = np.log1p(heatmap.astype(np.float64))
            range_max = min(
                self.args.max_range,
                (heatmap.shape[0] - 1) * self.args.range_resolution,
            )
            max_row = min(
                heatmap.shape[0],
                int(self.args.max_range / self.args.range_resolution) + 1,
            )
            min_row = min(
                max_row - 1,
                max(0, int(self.args.min_range / self.args.range_resolution)),
            )
            visible = image[:max_row]
            self.depth_image.set_data(visible)
            self.depth_image.set_extent((-90, 90, 0, range_max))
            finite = visible[np.isfinite(visible)]
            if finite.size:
                low, high = np.percentile(finite, (10, 99.5))
                if high > low:
                    self.depth_image.set_clim(low, high)

            candidates = heatmap[min_row:max_row]
            peak_rows = np.argmax(candidates, axis=0) + min_row
            depth = peak_rows.astype(float) * self.args.range_resolution
            spatial_frequency = (
                np.arange(-self.args.azimuth_bins // 2, self.args.azimuth_bins // 2)
                * (2.0 / self.args.azimuth_bins)
            )
            angles = np.degrees(np.arcsin(spatial_frequency))
            self.depth_line.set_data(angles, depth)

    def map_key(self, event):
        if event.key in ("r", "R") and self.environment is not None:
            self.environment.clear()
            self.map_image.set_data(np.ma.masked_all(self.environment.evidence.shape))
            self.figure.canvas.draw_idle()

    def close(self, _event=None):
        self.stop_reader.set()
        self.reader.join(timeout=1.0)
        if self.serial.is_open:
            self.serial.close()
        if self.record and not self.record.closed:
            self.record.close()

    def run(self):
        try:
            plt.show()
        finally:
            self.close()


def main():
    args = build_parser().parse_args()
    if args.limit <= 0:
        raise SystemExit("--limit must be positive")
    if args.fps <= 0 or args.map_history <= 0:
        raise SystemExit("--fps and --map-history must be positive")
    if args.azimuth_bins <= 0:
        raise SystemExit("--azimuth-bins must be positive")
    if args.range_resolution <= 0:
        raise SystemExit("--range-resolution must be positive")
    if args.min_range < 0 or args.max_range <= args.min_range:
        raise SystemExit("require 0 <= --min-range < --max-range")
    LiveViewer(args).run()


if __name__ == "__main__":
    main()
