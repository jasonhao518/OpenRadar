# Copyright 2026 The OpenRadar Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");

"""TI mmWave UART TLV parsing, including IWRL6432 ADC and point clouds."""

from __future__ import annotations

import io
import os
import struct
from dataclasses import dataclass
from typing import Dict, Iterator, List, Mapping, Tuple, Union

import numpy as np


TI_UART_MAGIC = bytes.fromhex("02 01 04 03 06 05 08 07")
TI_UART_HEADER = struct.Struct("<8s8I")
TI_UART_TLV_HEADER = struct.Struct("<II")
TI_UART_CAPTURE_METADATA = struct.Struct("<I8HI")
TI_UART_POINT_UNITS = struct.Struct("<4f2H")
TI_UART_COMPRESSED_POINT = struct.Struct("<hhhhBB")

TI_TLV_DETECTED_POINTS = 301
TI_TLV_RANGE_PROFILE_MAJOR = 302
TI_TLV_RANGE_PROFILE_MINOR = 303
TI_TLV_STATS = 306
TI_TLV_ADC_SAMPLES = 316
TI_TLV_CAPTURE_METADATA = 1100

_MAX_PACKET_SIZE = 16 * 1024 * 1024


@dataclass(frozen=True)
class TIUARTHeader:
    version: int
    total_packet_length: int
    platform: int
    frame_number: int
    cpu_cycles: int
    num_detected_objects: int
    num_tlvs: int
    subframe_number: int


@dataclass(frozen=True)
class TIUARTCaptureMetadata:
    version: int
    mode: int
    sample_format: int
    num_chirps: int
    num_rx_antennas: int
    num_tx_antennas: int
    num_adc_samples: int
    num_range_bins: int
    data_size_bytes: int


@dataclass(frozen=True)
class TIUARTFrame:
    header: TIUARTHeader
    tlvs: Mapping[int, Tuple[bytes, ...]]
    raw_packet: bytes

    def payloads(self, tlv_type: int) -> Tuple[bytes, ...]:
        return self.tlvs.get(tlv_type, ())

    def payload(self, tlv_type: int) -> bytes:
        payloads = self.payloads(tlv_type)
        if len(payloads) != 1:
            raise ValueError(
                "expected one TLV type {}, found {}".format(tlv_type, len(payloads))
            )
        return payloads[0]

    def range_profile(self, tlv_type: int = TI_TLV_RANGE_PROFILE_MAJOR) -> np.ndarray:
        payload = self.payload(tlv_type)
        if len(payload) % 4:
            raise ValueError("range-profile payload is not uint32 aligned")
        return np.frombuffer(payload, dtype="<u4").copy()

    def point_cloud(self) -> np.ndarray:
        """Return compressed TLV 301 as x/y/z/velocity/SNR/noise floats."""
        payload = self.payload(TI_TLV_DETECTED_POINTS)
        if len(payload) < TI_UART_POINT_UNITS.size:
            raise ValueError("compressed point-cloud payload is too short")
        xyz_unit, doppler_unit, snr_unit, noise_unit, major_count, minor_count = (
            TI_UART_POINT_UNITS.unpack_from(payload)
        )
        count = major_count + minor_count
        expected = TI_UART_POINT_UNITS.size + count * TI_UART_COMPRESSED_POINT.size
        if len(payload) != expected:
            raise ValueError(
                "point-cloud size mismatch: counts imply {}, payload is {}".format(
                    expected, len(payload)
                )
            )
        dtype = np.dtype(
            [
                ("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                ("velocity", "<f4"), ("snr", "<f4"), ("noise", "<f4"),
                ("mode", "u1"),
            ]
        )
        points = np.zeros(count, dtype=dtype)
        cursor = TI_UART_POINT_UNITS.size
        for index in range(count):
            x, y, z, velocity, snr, noise = TI_UART_COMPRESSED_POINT.unpack_from(
                payload, cursor
            )
            cursor += TI_UART_COMPRESSED_POINT.size
            points[index] = (
                x * xyz_unit, y * xyz_unit, z * xyz_unit,
                velocity * doppler_unit, snr * snr_unit, noise * noise_unit,
                0 if index < major_count else 1,
            )
        return points

    def capture_metadata(self) -> TIUARTCaptureMetadata:
        payload = self.payload(TI_TLV_CAPTURE_METADATA)
        if len(payload) != TI_UART_CAPTURE_METADATA.size:
            raise ValueError("invalid IWRL6432 capture metadata size")
        values = TI_UART_CAPTURE_METADATA.unpack(payload)
        return TIUARTCaptureMetadata(
            version=values[0], mode=values[1], sample_format=values[2],
            num_chirps=values[3], num_rx_antennas=values[4],
            num_tx_antennas=values[5], num_adc_samples=values[6],
            num_range_bins=values[7], data_size_bytes=values[9],
        )

    def adc_samples(self) -> np.ndarray:
        """Return signed-real16 ADC as [chirp, rx, sample]."""
        metadata = self.capture_metadata()
        payload = self.payload(TI_TLV_ADC_SAMPLES)
        if metadata.sample_format != 1:
            raise ValueError("only signed-real16 ADC sample format 1 is supported")
        shape = (
            metadata.num_chirps,
            metadata.num_rx_antennas,
            metadata.num_adc_samples,
        )
        expected = int(np.prod(shape)) * 2
        if metadata.data_size_bytes != len(payload) or expected != len(payload):
            raise ValueError("ADC payload size does not match metadata shape")
        return np.frombuffer(payload, dtype="<i2").reshape(shape).copy()


Source = Union[str, os.PathLike, bytes, bytearray, memoryview, io.BufferedIOBase]


def _read_source(source: Source) -> bytes:
    if isinstance(source, (bytes, bytearray, memoryview)):
        return bytes(source)
    if hasattr(source, "read"):
        return source.read()
    with open(source, "rb") as stream:
        return stream.read()


def iter_ti_uart_frames(source: Source, strict: bool = False) -> Iterator[TIUARTFrame]:
    """Yield complete packets, resynchronizing on the TI magic word."""
    data = _read_source(source)
    offset = 0
    while True:
        start = data.find(TI_UART_MAGIC, offset)
        if start < 0:
            return
        if start + TI_UART_HEADER.size > len(data):
            if strict:
                raise ValueError("truncated TI UART header")
            return
        values = TI_UART_HEADER.unpack_from(data, start)
        total_length = values[2]
        if total_length < TI_UART_HEADER.size or total_length > _MAX_PACKET_SIZE:
            if strict:
                raise ValueError("invalid TI UART packet length")
            offset = start + 1
            continue
        packet_end = start + total_length
        if packet_end > len(data):
            if strict:
                raise ValueError("truncated TI UART packet")
            return

        header = TIUARTHeader(*values[1:])
        cursor = start + TI_UART_HEADER.size
        decoded: Dict[int, List[bytes]] = {}
        malformed = False
        for _ in range(header.num_tlvs):
            if cursor + TI_UART_TLV_HEADER.size > packet_end:
                malformed = True
                break
            tlv_type, payload_length = TI_UART_TLV_HEADER.unpack_from(data, cursor)
            cursor += TI_UART_TLV_HEADER.size
            payload_end = cursor + payload_length
            if payload_end > packet_end:
                malformed = True
                break
            decoded.setdefault(tlv_type, []).append(data[cursor:payload_end])
            cursor = payload_end
        if malformed:
            if strict:
                raise ValueError("malformed TLV payload")
            offset = start + 1
            continue
        yield TIUARTFrame(
            header,
            {key: tuple(items) for key, items in decoded.items()},
            data[start:packet_end],
        )
        offset = packet_end


def load_ti_uart_frames(source: Source, strict: bool = False) -> List[TIUARTFrame]:
    return list(iter_ti_uart_frames(source, strict=strict))


class TIUARTStreamParser:
    """Incrementally decode arbitrarily chunked TI UART byte streams."""

    def __init__(self):
        self._buffer = bytearray()

    @property
    def buffered_bytes(self) -> int:
        return len(self._buffer)

    def reset(self) -> None:
        self._buffer.clear()

    def feed(self, data: Union[bytes, bytearray, memoryview]) -> List[TIUARTFrame]:
        if data:
            self._buffer.extend(data)
        frames: List[TIUARTFrame] = []
        while True:
            start = self._buffer.find(TI_UART_MAGIC)
            if start < 0:
                # Keep a possible partial magic word spanning two serial reads.
                keep = min(len(self._buffer), len(TI_UART_MAGIC) - 1)
                if len(self._buffer) > keep:
                    del self._buffer[:-keep]
                break
            if start:
                del self._buffer[:start]
            if len(self._buffer) < TI_UART_HEADER.size:
                break
            values = TI_UART_HEADER.unpack_from(self._buffer)
            total_length = values[2]
            if total_length < TI_UART_HEADER.size or total_length > _MAX_PACKET_SIZE:
                del self._buffer[0]
                continue
            if len(self._buffer) < total_length:
                break
            packet = bytes(self._buffer[:total_length])
            del self._buffer[:total_length]
            parsed = load_ti_uart_frames(packet, strict=True)
            if len(parsed) != 1:
                raise ValueError("complete serial packet did not decode to one frame")
            frames.append(parsed[0])
        return frames


def load_range_profiles(
    source: Source, tlv_type: int = TI_TLV_RANGE_PROFILE_MAJOR
) -> Tuple[np.ndarray, np.ndarray]:
    selected = [f for f in load_ti_uart_frames(source) if f.payloads(tlv_type)]
    if not selected:
        raise ValueError("no range-profile TLV {} found".format(tlv_type))
    profiles = [frame.range_profile(tlv_type) for frame in selected]
    if any(len(profile) != len(profiles[0]) for profile in profiles):
        raise ValueError("range-profile width changes between frames")
    return (
        np.asarray([f.header.frame_number for f in selected], dtype=np.uint32),
        np.stack(profiles),
    )


def load_adc_frames(source: Source) -> Tuple[np.ndarray, List[np.ndarray]]:
    selected = [
        f for f in load_ti_uart_frames(source)
        if f.payloads(TI_TLV_CAPTURE_METADATA) and f.payloads(TI_TLV_ADC_SAMPLES)
    ]
    if not selected:
        raise ValueError("no IWRL6432 metadata + ADC TLV frames found")
    return (
        np.asarray([f.header.frame_number for f in selected], dtype=np.uint32),
        [f.adc_samples() for f in selected],
    )
