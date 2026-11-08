#!/usr/bin/env python3
"""Live TI UART point-cloud and range-profile viewer."""

import argparse
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import serial
from matplotlib.animation import FuncAnimation

from mmwave.dataloader.ti_uart import (
    TI_TLV_DETECTED_POINTS,
    TI_TLV_RANGE_PROFILE_MAJOR,
    TIUARTStreamParser,
)


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
    parser.add_argument("--view", choices=("points", "range", "both"), default="both")
    parser.add_argument("--limit", type=float, default=5.0, help="point-cloud axis limit in metres")
    parser.add_argument("--interval", type=int, default=50, help="UI refresh interval in ms")
    parser.add_argument("--record", type=Path, help="also save received bytes to this file")
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

        if args.view == "both":
            self.figure, (self.point_axis, self.range_axis) = plt.subplots(1, 2)
        elif args.view == "points":
            self.figure, self.point_axis = plt.subplots()
            self.range_axis = None
        else:
            self.figure, self.range_axis = plt.subplots()
            self.point_axis = None
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
        return "{} @ {:,} baud  |  frame {}  |  {:,} bytes  |  {:,} packets".format(
            self.args.port,
            self.serial.baudrate,
            frame,
            self.bytes_received,
            self.frames_received,
        )

    def poll(self, _tick):
        try:
            waiting = self.serial.in_waiting
            chunk = self.serial.read(min(max(waiting, 1), 65536))
            if chunk:
                self.bytes_received += len(chunk)
                if self.record:
                    self.record.write(chunk)
                    self.record.flush()
                for frame in self.parser.feed(chunk):
                    self.update_frame(frame)
        except (OSError, serial.SerialException, ValueError) as error:
            self.error = error
        self.status.set_text(self._status_text())
        return self.artists

    @property
    def artists(self):
        return tuple(
            artist for artist in (self.scatter, self.range_line, self.status)
            if artist is not None
        )

    def update_frame(self, frame):
        self.frames_received += 1
        self.last_frame = frame.header.frame_number

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

    def close(self, _event=None):
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
    LiveViewer(args).run()


if __name__ == "__main__":
    main()
