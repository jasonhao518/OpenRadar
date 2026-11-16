#!/usr/bin/env python3
"""Live TI UART point-cloud, range-profile, depth-map and people viewer."""

import argparse
import time
import queue
import threading
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import serial
from matplotlib.animation import FuncAnimation
from matplotlib.colors import LinearSegmentedColormap

from mmwave.dataloader.ti_uart import (
    TI_TLV_DETECTED_POINTS,
    TI_TLV_RANGE_AZIMUTH_HEATMAP_MAJOR,
    TI_TLV_RANGE_PROFILE_MAJOR,
    TIUARTStreamParser,
    load_ti_uart_frames,
)
from mmwave.dsp.environment_map import ReflectionMap
from mmwave.tracking.people import PeopleDetector

# Mark colours: a blue/red diverging pair with a neutral midpoint for signed
# radial velocity, and fixed hues for the two kinds of track.
INK = "#0b0b0b"
INK_MUTED = "#52514e"
GRID = "#d9d8d3"
PERSON = "#eb6834"
FIRMWARE_TRACK = "#1baf7a"
VELOCITY_CMAP = LinearSegmentedColormap.from_list(
    "radial_velocity", ["#2a78d6", "#c9c8c3", "#e34948"])
DEFAULT_FPS = 4.0


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", help="serial device (not needed with --replay)")
    parser.add_argument(
        "--replay", type=Path,
        help="play back a capture written by --record instead of opening a port",
    )
    parser.add_argument(
        "--replay-speed", type=float, default=1.0,
        help="playback rate relative to real time (default: 1)",
    )
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
        "--view", choices=("points", "range", "both", "depth", "depth3d", "map", "scene", "people", "all"),
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
    parser.add_argument("--map-history", type=float, default=30.0, help="sliding map history window in seconds")
    parser.add_argument("--map-threshold", type=float, default=6.0, help="peak contrast above median background in dB")
    parser.add_argument(
        "--fps", type=float,
        help="radar frame rate; default: taken from --config, else measured "
             "live, else {:g}".format(DEFAULT_FPS),
    )
    people = parser.add_argument_group("people view")
    people.add_argument(
        "--static-filter", choices=("auto", "on", "off"), default="auto",
        help="learn and hide persistent reflectors on the host. Needed only "
             "when the radar runs with clutterRemoval 0 (default: auto)",
    )
    people.add_argument("--people-min-snr", type=float, default=10.0, help="ignore weaker points (dB)")
    people.add_argument("--people-eps", type=float, default=0.6, help="cluster radius in metres")
    people.add_argument("--people-min-points", type=int, default=3, help="points needed to form a cluster")
    people.add_argument("--people-window", type=float, default=0.5, help="seconds of points pooled per cluster step")
    people.add_argument("--people-hold", type=float, default=3.0, help="seconds a person is kept after the last detection")
    return parser


def read_config_settings(config_path):
    """Pick the few values the host needs out of a TI .cfg file."""
    settings = {}
    for line in config_path.read_text(encoding="utf-8").splitlines():
        words = line.split()
        if not words or words[0].startswith(("%", "#")):
            continue
        try:
            if words[0] == "frameCfg" and float(words[5]) > 0:
                settings["fps"] = 1000.0 / float(words[5])
            elif words[0] == "clutterRemoval":
                settings["clutter_removal"] = int(words[1]) != 0
        except (IndexError, ValueError):
            pass
    return settings


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
        settings = read_config_settings(args.config) if args.config else {}
        self.fps = args.fps or settings.get("fps") or DEFAULT_FPS
        # Measure the frame rate from arrival times only when nothing states it.
        self.fps_known = bool(args.fps or settings.get("fps") or args.replay)
        self.fps_probe = None
        self.serial = None
        if not args.replay:
            initial_baud = args.cli_baud if args.config else args.baud
            self.serial = serial.Serial(args.port, initial_baud, timeout=0)
            self.serial.reset_input_buffer()
            if args.config:
                send_config(self.serial, args.config, args.command_timeout)
        self.parser = TIUARTStreamParser()
        self.preamble = bytearray()
        self.record = args.record.open("wb") if args.record else None
        self.bytes_received = 0
        self.frames_received = 0
        self.last_frame = None
        self.error = None
        self.skipped_payloads = 0
        self.pending = queue.Queue()
        self.stop_reader = threading.Event()
        if self.serial is not None:
            self.serial.timeout = 0.05
        self.reader = threading.Thread(
            target=self.replay if args.replay else self.receive, daemon=True)

        show_points = args.view in ("points", "both", "all")
        show_range = args.view in ("range", "both", "all")
        show_depth = args.view in ("depth", "depth3d", "scene", "all")
        show_3d = args.view in ("depth3d", "scene", "all")
        show_map = args.view in ("map", "scene", "all")
        show_people = args.view == "people"
        panel_count = sum((show_points, show_range, show_depth, show_3d, show_map))
        if show_people:
            self.figure = plt.figure(figsize=(9, 8))
            self.people_axis = self.figure.add_subplot(1, 1, 1)
            panel_count = 0
        else:
            self.people_axis = None
        if not show_people:
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
                range_step=args.range_resolution, fps=self.fps,
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
                extent=(-1, 1, 0, args.max_range),
            )
            (self.depth_line,) = self.depth_axis.plot(
                [], [], color="white", linewidth=1.5, label="strongest return"
            )
            self.depth_axis.set(
                xlabel="Azimuth angle (deg)", ylabel="Range / depth (m)",
                xlim=(-1, 1), ylim=(0, args.max_range),
                title="Dense range-azimuth depth view (TLV 304)",
            )
            # Azimuth FFT bins are uniform in sin(angle), not in angle.
            ticks = np.array([-90, -60, -30, -15, 0, 15, 30, 60, 90])
            self.depth_axis.set_xticks(np.sin(np.radians(ticks)))
            self.depth_axis.set_xticklabels([str(tick) for tick in ticks])
            self.depth_axis.legend(loc="upper right")
            colorbar = self.figure.colorbar(
                self.depth_image, ax=self.depth_axis, pad=0.02
            )
            colorbar.set_label("log(1 + return intensity)")

        self.detector = None
        if self.people_axis is not None:
            self._build_people_panel(settings)

        self.status = self.figure.suptitle(self._status_text(), fontsize=9, color=INK_MUTED)
        self.figure.tight_layout(rect=(0, 0, 1, 0.94))
        self.reader.start()
        self.animation = FuncAnimation(
            self.figure,
            self.poll,
            interval=args.interval,
            blit=False,
            cache_frame_data=False,
        )

    def _build_people_panel(self, settings):
        args = self.args
        axis = self.people_axis
        static_filter = {"auto": None, "on": True, "off": False}[args.static_filter]
        if static_filter is None and "clutter_removal" in settings:
            static_filter = not settings["clutter_removal"]
        self.detector = PeopleDetector(
            min_range=args.min_range, max_range=args.max_range,
            min_snr=args.people_min_snr, window=args.people_window,
            eps=args.people_eps, min_points=args.people_min_points,
            hold=args.people_hold, static_filter=static_filter,
        )
        self.people_clock = 0.0
        self.people_last_frame = None
        self.people_ids = ()
        self.firmware_tracks_seen = False

        for radius in np.arange(1.0, args.max_range + 0.01, 1.0):
            arc = np.radians(np.linspace(-90, 90, 91))
            axis.plot(radius * np.sin(arc), radius * np.cos(arc), color=GRID, linewidth=0.8, zorder=0)
            axis.text(0.0, radius, "{:g} m".format(radius), color=INK_MUTED, fontsize=7,
                      ha="center", va="bottom", zorder=0)
        for bearing in (-60, -30, 30, 60):
            angle = np.radians(bearing)
            axis.plot([0, args.max_range * np.sin(angle)], [0, args.max_range * np.cos(angle)],
                      color=GRID, linewidth=0.8, zorder=0)
        axis.plot(0, 0, marker="^", color=INK, markersize=9, linestyle="none", zorder=5)

        self.static_scatter = axis.scatter(
            [], [], marker="x", s=14, linewidths=0.8, color="#a9a8a2",
            label="static reflection (ignored)", zorder=1)
        self.mover_scatter = axis.scatter(
            [], [], c=[], s=22, cmap=VELOCITY_CMAP, vmin=-2, vmax=2,
            edgecolors=INK_MUTED, linewidths=0.3, label="moving point", zorder=2)
        self.person_scatter = axis.scatter(
            [], [], s=420, facecolors="none", edgecolors=PERSON, linewidths=2.2,
            label="person", zorder=4)
        self.lost_scatter = axis.scatter(
            [], [], s=420, facecolors="none", edgecolors=PERSON, linewidths=1.4,
            linestyles="dotted", label="person, last seen here", zorder=4)
        self.target_scatter = axis.scatter(
            [], [], marker="s", s=300, facecolors="none", edgecolors=FIRMWARE_TRACK,
            linewidths=1.8, label="radar's own track (TLV 308)", zorder=3)
        self.person_labels = []
        self.person_trails = []
        self.target_labels = []
        self.figure.colorbar(self.mover_scatter, ax=axis, pad=0.02, shrink=0.8).set_label(
            "Radial velocity of moving points (m/s)")
        axis.set(
            xlabel="x / lateral (m)", ylabel="y / forward (m)",
            xlim=(-args.limit, args.limit), ylim=(-0.3, args.max_range),
        )
        axis.set_aspect("equal", adjustable="box")
        for side in ("top", "right"):
            axis.spines[side].set_visible(False)
        axis.legend(loc="lower right", fontsize=7, frameon=False, markerscale=0.4)
        self.people_title = axis.set_title(
            "Waiting for radar frames", fontsize=14, color=INK, loc="left", pad=18)
        self.people_note = axis.text(
            0.0, 1.01, "", transform=axis.transAxes, fontsize=8, color=INK_MUTED, va="bottom")
        self.figure.canvas.mpl_connect("key_press_event", self.people_key)

    def _status_text(self):
        if self.error:
            return "Radar error: {}".format(self.error)
        frame = "waiting" if self.last_frame is None else str(self.last_frame)
        if self.serial is None:
            source = "replay {} @ {:g} FPS".format(self.args.replay.name, self.fps)
        else:
            source = "{} @ {:,} baud".format(self.args.port, self.serial.baudrate)
        return "{}  |  frame {}  |  {:,} bytes  |  {:,} packets | skipped {}".format(
            source,
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
                    if self.frames_received == 0 and len(self.preamble) < 4096:
                        self.preamble.extend(chunk[:4096])
                    if self.record:
                        self.record.write(chunk)
                        self.record.flush()
                    self.pending.put(chunk)
        except (OSError, serial.SerialException) as error:
            if not self.stop_reader.is_set():
                self.error = error

    def replay(self):
        """Feed a recorded capture at the radar's frame rate."""
        try:
            frames = load_ti_uart_frames(self.args.replay)
            if not frames:
                raise ValueError("no TI UART packets in {}".format(self.args.replay))
            period = 1.0 / (self.fps * self.args.replay_speed)
            for frame in frames:
                if self.stop_reader.wait(period):
                    return
                self.bytes_received += len(frame.raw_packet)
                self.pending.put(frame.raw_packet)
        except (OSError, ValueError) as error:
            self.error = error

    def _cli_error(self):
        """Text the radar printed instead of starting, e.g. a rejected sensorStart."""
        text = bytes(self.preamble).decode("ascii", errors="replace")
        for line in text.splitlines():
            if "Error" in line:
                return line.strip()
        return None

    def _measure_fps(self, frame_number):
        now = time.monotonic()
        if self.fps_probe is None:
            self.fps_probe = (now, frame_number)
            return
        start, first = self.fps_probe
        if now - start >= 3.0 and frame_number > first:
            self.fps = (frame_number - first) / (now - start)
            self.fps_known = True
            print("Measured radar frame rate: {:.1f} FPS".format(self.fps), flush=True)

    def poll(self, _tick):
        if self.frames_received == 0 and self.error is None and self.preamble:
            self.error = self._cli_error()
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
                getattr(self, "mover_scatter", None),
                getattr(self, "person_scatter", None),
            )
            if artist is not None
        )

    def update_frame(self, frame):
        self.frames_received += 1
        self.last_frame = frame.header.frame_number
        if not self.fps_known:
            self._measure_fps(frame.header.frame_number)
        if self.detector is not None:
            self.update_people(frame)

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
            # The firmware already centres broadside at column N/2; shifting
            # again would move it to the +/-90 degree edges.
            heatmap = frame.range_azimuth_heatmap(self.args.azimuth_bins)
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
            step = 2.0 / self.args.azimuth_bins
            self.depth_image.set_extent(
                (-1 - step / 2, 1 - step / 2, 0, range_max))
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
            self.depth_line.set_data(spatial_frequency, depth)

    def update_people(self, frame):
        number = frame.header.frame_number
        if self.people_last_frame is None or number <= self.people_last_frame:
            self.detector.reset()
            self.people_clock = 0.0
        else:
            self.people_clock += (number - self.people_last_frame) / self.fps
        self.people_last_frame = number

        if frame.payloads(TI_TLV_DETECTED_POINTS):
            points = frame.point_cloud()
        else:
            points = np.empty((0, 4))
        people = self.detector.update(points, self.people_clock)

        movers = self.detector.movers
        self.mover_scatter.set_offsets(movers[:, :2].reshape(-1, 2))
        self.mover_scatter.set_array(movers[:, 2])
        self.static_scatter.set_offsets(self.detector.static_points.reshape(-1, 2))

        seen = [person for person in people if not person.coasting]
        lost = [person for person in people if person.coasting]
        self.person_scatter.set_offsets(np.array([(p.x, p.y) for p in seen]).reshape(-1, 2))
        self.lost_scatter.set_offsets(np.array([(p.x, p.y) for p in lost]).reshape(-1, 2))
        self._fit_pool(self.person_labels, len(people), self._new_label)
        self._fit_pool(self.person_trails, len(people), self._new_trail)
        for person, label, trail in zip(people, self.person_labels, self.person_trails):
            label.set_position((person.x + 0.3, person.y + 0.15))
            label.set_text("P{}  {:.1f} m  {:+.0f}\N{DEGREE SIGN}\n{}".format(
                person.id, person.range, person.azimuth_deg,
                "last seen {:.0f} s ago".format(person.since_seen) if person.coasting
                else "{:.1f} m/s".format(person.speed)))
            path = np.array(person.trail).reshape(-1, 2)
            trail.set_data(path[:, 0], path[:, 1])

        targets = frame.targets()
        if len(targets):
            self.firmware_tracks_seen = True
        self.target_scatter.set_offsets(np.column_stack((targets["x"], targets["y"])))
        self._fit_pool(self.target_labels, len(targets), self._new_label)
        for target, label in zip(targets, self.target_labels):
            label.set_position((float(target["x"]) - 0.3, float(target["y"]) - 0.15))
            label.set_ha("right")
            label.set_va("top")
            label.set_text("T{}".format(int(target["tid"])))

        count = len(people)
        title = "{} {} in view".format(
            "No" if count == 0 else count, "person" if count == 1 else "people")
        if self.firmware_tracks_seen:
            title += "   (radar's own tracker: {})".format(len(targets))
        self.people_title.set_text(title)
        self.people_note.set_text(
            "Radar is not removing static clutter: hiding persistent reflections "
            "on the host (press R to relearn the room)"
            if self.detector.static_filter_enabled else "")

        ids = tuple(person.id for person in people)
        if ids != self.people_ids:
            self.people_ids = ids
            print("frame {}: {}".format(number, ", ".join(
                "P{} at {:.1f} m {:+.0f} deg".format(p.id, p.range, p.azimuth_deg)
                for p in people) or "nobody"), flush=True)

    def _new_label(self):
        return self.people_axis.text(
            0, 0, "", color=INK, fontsize=8, va="bottom", zorder=6,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.7, pad=1.0))

    def _new_trail(self):
        (line,) = self.people_axis.plot([], [], color=PERSON, linewidth=1.2, alpha=0.6, zorder=3)
        return line

    @staticmethod
    def _fit_pool(pool, count, factory):
        while len(pool) < count:
            pool.append(factory())
        for index, artist in enumerate(pool):
            artist.set_visible(index < count)

    def people_key(self, event):
        if event.key in ("r", "R") and self.detector is not None:
            self.detector.reset()
            self.people_last_frame = None

    def map_key(self, event):
        if event.key in ("r", "R") and self.environment is not None:
            self.environment.clear()
            self.map_image.set_data(np.ma.masked_all(self.environment.evidence.shape))
            self.figure.canvas.draw_idle()

    def close(self, _event=None):
        self.stop_reader.set()
        self.reader.join(timeout=1.0)
        if self.serial is not None and self.serial.is_open:
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
    if bool(args.port) == bool(args.replay):
        raise SystemExit("give exactly one of --port or --replay")
    if args.replay and args.config:
        raise SystemExit("--config cannot be sent during --replay; use --fps if needed")
    if (args.fps is not None and args.fps <= 0) or args.map_history <= 0 or args.replay_speed <= 0:
        raise SystemExit("--fps, --map-history and --replay-speed must be positive")
    if args.azimuth_bins <= 0:
        raise SystemExit("--azimuth-bins must be positive")
    if args.range_resolution <= 0:
        raise SystemExit("--range-resolution must be positive")
    if args.min_range < 0 or args.max_range <= args.min_range:
        raise SystemExit("require 0 <= --min-range < --max-range")
    LiveViewer(args).run()


if __name__ == "__main__":
    main()
