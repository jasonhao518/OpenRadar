import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np

MODULE_PATH = Path(__file__).parents[1] / "dataloader" / "ti_uart.py"
SPEC = importlib.util.spec_from_file_location("openradar_ti_uart", MODULE_PATH)
ti_uart = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = ti_uart
SPEC.loader.exec_module(ti_uart)

from openradar_ti_uart import (
    TI_TLV_ADC_SAMPLES,
    TI_TLV_CAPTURE_METADATA,
    TI_TLV_DETECTED_POINTS,
    TI_TLV_RANGE_PROFILE_MAJOR,
    TI_UART_CAPTURE_METADATA,
    TI_UART_HEADER,
    TI_UART_MAGIC,
    TI_UART_POINT_UNITS,
    TI_UART_COMPRESSED_POINT,
    TI_UART_TLV_HEADER,
    load_adc_frames,
    load_range_profiles,
    load_ti_uart_frames,
)


def make_packet(frame_number, tlvs, object_count=0):
    body = b"".join(
        TI_UART_TLV_HEADER.pack(tlv_type, len(payload)) + payload
        for tlv_type, payload in tlvs
    )
    unpadded = TI_UART_HEADER.size + len(body)
    total = (unpadded + 31) // 32 * 32
    header = TI_UART_HEADER.pack(
        TI_UART_MAGIC, 0x01000000, total, 0xA6432, frame_number,
        0, object_count, len(tlvs), 0,
    )
    return header + body + bytes(total - unpadded)


class TIUARTTest(unittest.TestCase):
    def test_range_profiles_and_resynchronization(self):
        first = np.array([10, 20, 30, 40], dtype="<u4")
        second = np.array([11, 21, 31, 41], dtype="<u4")
        capture = (
            b"CLI text before magic\n"
            + make_packet(7, [(TI_TLV_RANGE_PROFILE_MAJOR, first.tobytes())])
            + make_packet(8, [(TI_TLV_RANGE_PROFILE_MAJOR, second.tobytes())])
        )
        numbers, profiles = load_range_profiles(capture)
        np.testing.assert_array_equal(numbers, [7, 8])
        np.testing.assert_array_equal(profiles, [first, second])

    def test_compressed_point_cloud(self):
        payload = TI_UART_POINT_UNITS.pack(0.01, 0.1, 0.25, 1.0, 1, 1)
        payload += TI_UART_COMPRESSED_POINT.pack(100, 200, -50, 3, 20, 4)
        payload += TI_UART_COMPRESSED_POINT.pack(-100, 50, 25, -2, 12, 8)
        frame = load_ti_uart_frames(
            make_packet(9, [(TI_TLV_DETECTED_POINTS, payload)], object_count=2)
        )[0]
        points = frame.point_cloud()
        np.testing.assert_allclose(points["x"], [1.0, -1.0])
        np.testing.assert_allclose(points["velocity"], [0.3, -0.2])
        np.testing.assert_array_equal(points["mode"], [0, 1])

    def test_iwrl6432_raw_adc_shape(self):
        adc = np.arange(-18, 18, dtype="<i2").reshape(2, 3, 6)
        metadata = TI_UART_CAPTURE_METADATA.pack(
            0x00010000, 1, 1, 2, 3, 2, 6, 8, 0, adc.nbytes
        )
        capture = make_packet(
            19,
            [
                (TI_TLV_CAPTURE_METADATA, metadata),
                (TI_TLV_ADC_SAMPLES, adc.tobytes()),
            ],
        )
        numbers, frames = load_adc_frames(capture)
        np.testing.assert_array_equal(numbers, [19])
        self.assertEqual(frames[0].shape, (2, 3, 6))
        np.testing.assert_array_equal(frames[0], adc)

    def test_strict_rejects_truncated_packet(self):
        packet = make_packet(1, [(TI_TLV_RANGE_PROFILE_MAJOR, b"\0" * 16)])
        with self.assertRaises(ValueError):
            load_ti_uart_frames(packet[:-1], strict=True)


if __name__ == "__main__":
    unittest.main()
