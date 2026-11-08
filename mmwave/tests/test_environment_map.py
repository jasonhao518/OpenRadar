import importlib.util
from pathlib import Path
import unittest
import numpy as np

spec = importlib.util.spec_from_file_location(
    "environment_map", Path(__file__).parents[1] / "dsp" / "environment_map.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class MapTest(unittest.TestCase):
    def test_static_return_accumulates_then_fades(self):
        mapper = module.ReflectionMap(history=2, fps=4)
        power = np.ones((64, 32)) * 100
        power[10, 0] = 10000  # broadside, approximately 1.95 m
        for number in range(40):
            mapper.update(power, number)
        row, col = np.unravel_index(mapper.evidence.argmax(), mapper.evidence.shape)
        self.assertLess(abs(mapper.x[col]), 0.15)
        self.assertAlmostEqual(mapper.y[row], 1.95, delta=0.15)
        self.assertGreater(mapper.evidence.max(), 0.9)
        self.assertEqual(mapper.evidence[0, 0], 0)  # unobserved is not free
        for number in range(40, 100):
            mapper.update(np.ones((64, 32))*100, number)
        self.assertLess(mapper.evidence.max(), 0.01)
        mapper.clear()
        self.assertEqual(mapper.evidence.max(), 0)

    def test_positive_azimuth_maps_right(self):
        mapper = module.ReflectionMap(history=1)
        power = np.ones((64, 32))*100
        power[20, 8] = 10000  # sin(theta) = +0.5
        for number in range(20):
            mapper.update(power, number)
        row, col = np.unravel_index(mapper.evidence.argmax(), mapper.evidence.shape)
        self.assertGreater(mapper.x[col], 1.5)
        self.assertGreater(mapper.y[row], 3)


if __name__ == "__main__":
    unittest.main()
