import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np

SPEC = importlib.util.spec_from_file_location(
    "openradar_people", Path(__file__).parents[1] / "tracking" / "people.py")
people = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = people
SPEC.loader.exec_module(people)

FPS = 10.0


def cloud(rng, centre, count, velocity, spread=0.15):
    xy = rng.normal(centre, spread, size=(count, 2))
    return np.column_stack([xy, np.full(count, velocity), np.full(count, 20.0)])


class DbscanTest(unittest.TestCase):
    def test_two_groups_and_noise(self):
        points = np.array([[0, 1], [0.1, 1], [0, 1.1], [3, 3], [3.1, 3], [3, 3.1], [9, 9]])
        labels = people.dbscan(points, eps=0.3, min_points=3)
        self.assertEqual(len(set(labels[:3])), 1)
        self.assertEqual(len(set(labels[3:6])), 1)
        self.assertNotEqual(labels[0], labels[3])
        self.assertEqual(labels[6], -1)


class PeopleDetectorTest(unittest.TestCase):
    def test_walker_keeps_one_identity_and_position(self):
        rng = np.random.default_rng(1)
        detector = people.PeopleDetector(static_filter=False)
        ids = set()
        for frame in range(60):
            position = np.array([-1.5 + 0.05 * frame, 3.0])   # 0.5 m/s to the right
            points = np.vstack([
                cloud(rng, position, 5, 0.4),
                [[rng.uniform(-4, 4), rng.uniform(0.5, 7), 0.3, 12.0]],   # clutter
            ])
            found = detector.update(points, frame / FPS)
            ids.update(person.id for person in found)
        self.assertEqual(len(found), 1)
        self.assertEqual(len(ids), 1)
        self.assertAlmostEqual(found[0].x, position[0], delta=0.3)
        self.assertAlmostEqual(found[0].y, 3.0, delta=0.3)
        self.assertAlmostEqual(found[0].state[2], 0.5, delta=0.25)

    def test_two_people_and_hold_after_they_stop(self):
        rng = np.random.default_rng(2)
        detector = people.PeopleDetector(static_filter=False, hold=2.0)
        for frame in range(30):
            points = np.vstack([cloud(rng, (-1.0, 2.0), 4, 0.2),
                                cloud(rng, (1.5, 4.0), 4, -0.2)])
            found = detector.update(points, frame / FPS)
        self.assertEqual(len(found), 2)
        # Nobody moves: the radar reports no points.  Hold, then forget.
        found = detector.update(np.empty((0, 4)), 30 / FPS + 1.0)
        self.assertEqual(len(found), 2)
        self.assertTrue(all(person.coasting for person in found))
        found = detector.update(np.empty((0, 4)), 30 / FPS + 2.5)
        self.assertEqual(found, [])

    def test_static_filter_hides_walls_but_not_a_newcomer(self):
        rng = np.random.default_rng(3)
        detector = people.PeopleDetector(static_filter=True)
        wall = np.array([[x, 3.0, 0.0, 25.0] for x in (-1.0, -0.9, -0.8, 1.2, 1.3, 1.4)])
        for frame in range(100):
            jitter = np.column_stack([rng.normal(0, 0.03, (6, 2)), np.zeros((6, 2))])
            found = detector.update(wall + jitter, frame / FPS)
            self.assertEqual(found, [], "static wall reported as a person")
        for frame in range(100, 115):
            points = np.vstack([wall, cloud(rng, (0.2, 1.5), 4, 0.0)])
            found = detector.update(points, frame / FPS)
        self.assertEqual(len(found), 1)
        self.assertAlmostEqual(found[0].y, 1.5, delta=0.3)

    def test_auto_mode_stops_filtering_once_doppler_is_seen(self):
        detector = people.PeopleDetector()
        self.assertTrue(detector.static_filter_enabled)
        detector.update(np.array([[0.0, 2.0, 0.7, 20.0]]), 0.0)
        self.assertFalse(detector.static_filter_enabled)

    def test_stream_restart_resets_tracks(self):
        rng = np.random.default_rng(4)
        detector = people.PeopleDetector(static_filter=False)
        for frame in range(10):
            detector.update(cloud(rng, (0.0, 2.0), 5, 0.3), frame / FPS)
        self.assertEqual(len(detector.people), 1)
        detector.update(np.empty((0, 4)), 0.0)
        self.assertEqual(detector.people, [])


if __name__ == "__main__":
    unittest.main()
