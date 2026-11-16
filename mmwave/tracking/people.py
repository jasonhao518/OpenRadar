# Copyright 2026 The OpenRadar Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");

"""Turn sparse radar point clouds into tracked people.

The pipeline is deliberately small and only needs NumPy:

1. gate points by range and SNR;
2. optionally drop *static* reflectors (walls, furniture) by learning which
   map cells return a point in most frames -- only needed when the radar's
   own static clutter removal is disabled;
3. pool the remaining "mover" points over a short time window and group
   them with DBSCAN;
4. follow cluster centroids with a constant-velocity Kalman filter so that a
   person keeps one identity and survives a few missed frames.

Coordinates follow the TI convention: x is lateral (positive to the right
when looking out of the sensor), y is forward range, both in metres.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, List, Optional, Tuple

import numpy as np


def dbscan(points: np.ndarray, eps: float, min_points: int) -> np.ndarray:
    """Label 2-D points with cluster ids; noise is -1.  O(N^2), fine for N < 1e3."""
    count = len(points)
    labels = np.full(count, -1, dtype=int)
    if count == 0:
        return labels
    delta = points[:, None, :] - points[None, :, :]
    near = np.einsum("ijk,ijk->ij", delta, delta) <= eps * eps
    core = near.sum(axis=1) >= min_points
    cluster = 0
    for seed in range(count):
        if labels[seed] != -1 or not core[seed]:
            continue
        labels[seed] = cluster
        stack = [seed]
        while stack:
            index = stack.pop()
            if not core[index]:
                continue
            for neighbour in np.flatnonzero(near[index] & (labels == -1)):
                labels[neighbour] = cluster
                stack.append(neighbour)
        cluster += 1
    return labels


class StaticFilter:
    """Learn where reflections are persistent and flag those points as static."""

    def __init__(self, extent: float, cell: float = 0.25,
                 time_constant: float = 20.0, fraction: float = 0.2,
                 warmup: float = 1.0):
        self.extent = extent
        self.cell = cell
        self.time_constant = time_constant
        self.fraction = fraction
        self.warmup = warmup
        size_x = int(np.ceil(2 * extent / cell))
        size_y = int(np.ceil(extent / cell))
        self.occupancy = np.zeros((size_y, size_x))
        self.weight = 0.0
        self.elapsed = 0.0

    def reset(self) -> None:
        self.occupancy.fill(0)
        self.weight = 0.0
        self.elapsed = 0.0

    def _cells(self, xy: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        col = np.floor((xy[:, 0] + self.extent) / self.cell).astype(int)
        row = np.floor(xy[:, 1] / self.cell).astype(int)
        inside = ((row >= 0) & (row < self.occupancy.shape[0])
                  & (col >= 0) & (col < self.occupancy.shape[1]))
        return row, col, inside

    def update(self, xy: np.ndarray, dt: float) -> np.ndarray:
        """Return a boolean mask that is True for points judged static."""
        row, col, inside = self._cells(xy)
        hit = np.zeros_like(self.occupancy, dtype=bool)
        # A reflector jitters by about one cell between frames; dilate by one.
        rows, cols = self.occupancy.shape
        for d_row in (-1, 0, 1):
            for d_col in (-1, 0, 1):
                r = np.clip(row[inside] + d_row, 0, rows - 1)
                c = np.clip(col[inside] + d_col, 0, cols - 1)
                hit[r, c] = True
        alpha = min(1.0, dt / self.time_constant)
        self.occupancy += alpha * (hit - self.occupancy)
        self.weight += alpha * (1.0 - self.weight)
        self.elapsed += dt
        static = np.zeros(len(xy), dtype=bool)
        if self.elapsed < self.warmup:
            # Nothing is known yet: do not report the whole room as movers.
            static[:] = True
            return static
        level = self.occupancy / max(self.weight, 1e-9)
        static[inside] = level[row[inside], col[inside]] >= self.fraction
        return static


@dataclass
class Track:
    """One followed person.  ``coasting`` means no detection this frame."""

    id: int
    state: np.ndarray                      # x, y, vx, vy
    covariance: np.ndarray
    hits: int = 1
    age: float = 0.0
    since_seen: float = 0.0
    confirmed: bool = False
    num_points: int = 0
    radial_velocity: float = 0.0
    trail: Deque[Tuple[float, float]] = field(
        default_factory=lambda: deque(maxlen=50))

    @property
    def x(self) -> float:
        return float(self.state[0])

    @property
    def y(self) -> float:
        return float(self.state[1])

    @property
    def range(self) -> float:
        return float(np.hypot(self.state[0], self.state[1]))

    @property
    def azimuth_deg(self) -> float:
        return float(np.degrees(np.arctan2(self.state[0], self.state[1])))

    @property
    def speed(self) -> float:
        return float(np.hypot(self.state[2], self.state[3]))

    @property
    def coasting(self) -> bool:
        return self.since_seen > 0.0


class PeopleDetector:
    """Cluster mover points and track the clusters.  Call :meth:`update` per frame.

    static_filter: True to learn and remove persistent reflectors on the
        host, False when the radar already removes static clutter, or None to
        decide automatically (the filter stays on until a point with non-zero
        Doppler velocity is seen, which only clutter-removed/Doppler-enabled
        radar configurations produce).
    """

    def __init__(self, min_range: float = 0.3, max_range: float = 8.0,
                 min_snr: float = 10.0, window: float = 0.5,
                 eps: float = 0.6, min_points: int = 3,
                 static_filter: Optional[bool] = None,
                 static_time: float = 20.0, static_fraction: float = 0.2,
                 gate: float = 1.2, confirm_hits: int = 3,
                 tentative_timeout: float = 0.6, hold: float = 3.0,
                 measurement_sigma: float = 0.3, acceleration_sigma: float = 2.0):
        if min(max_range, window, eps, gate, hold) <= 0 or min_points < 1:
            raise ValueError("detector parameters must be positive")
        self.min_range = min_range
        self.max_range = max_range
        self.min_snr = min_snr
        self.window = window
        self.eps = eps
        self.min_points = min_points
        self.gate = gate
        self.confirm_hits = confirm_hits
        self.tentative_timeout = tentative_timeout
        self.hold = hold
        self.measurement_sigma = measurement_sigma
        self.acceleration_sigma = acceleration_sigma
        self._auto_static = static_filter is None
        self.static_filter_enabled = True if static_filter is None else bool(static_filter)
        self._static = StaticFilter(max_range, time_constant=static_time,
                                    fraction=static_fraction)
        self._recent: Deque[Tuple[float, np.ndarray]] = deque()
        self._next_id = 1
        self._time: Optional[float] = None
        self.tracks: List[Track] = []
        # Per-frame products, kept for display.
        self.movers = np.empty((0, 4))         # x, y, radial velocity, snr
        self.static_points = np.empty((0, 2))
        self.clusters = np.empty((0, 2))

    def reset(self) -> None:
        self._static.reset()
        self._recent.clear()
        self._time = None
        self.tracks = []
        self.movers = np.empty((0, 4))
        self.static_points = np.empty((0, 2))
        self.clusters = np.empty((0, 2))

    @property
    def people(self) -> List[Track]:
        return [track for track in self.tracks if track.confirmed]

    # ------------------------------------------------------------------ #

    def update(self, points, timestamp: float) -> List[Track]:
        """Consume one frame.

        points: structured array with x, y, velocity, snr fields (as returned
            by ``TIUARTFrame.point_cloud``) or an (N, 4) float array in that
            order.  Pass an empty array for a frame without detections.
        timestamp: frame time in seconds (frame number / frame rate).
        """
        if self._time is not None and timestamp <= self._time:
            self.reset()            # stream restarted
        dt = 0.0 if self._time is None else timestamp - self._time
        self._time = timestamp

        data = self._as_array(points)
        ranges = np.hypot(data[:, 0], data[:, 1])
        keep = ((ranges >= self.min_range) & (ranges <= self.max_range)
                & (data[:, 1] > 0) & (data[:, 3] >= self.min_snr))
        data = data[keep]

        if self._auto_static and self.static_filter_enabled and np.any(data[:, 2] != 0):
            self.static_filter_enabled = False
        if self.static_filter_enabled:
            static = self._static.update(data[:, :2], dt if dt > 0 else 0.1)
        else:
            static = np.zeros(len(data), dtype=bool)
        self.static_points = data[static, :2]
        self.movers = data[~static]

        self._recent.append((timestamp, self.movers))
        while self._recent and self._recent[0][0] <= timestamp - self.window:
            self._recent.popleft()
        pooled = np.concatenate([item[1] for item in self._recent])

        detections = self._cluster(pooled)
        self.clusters = np.array([d[0] for d in detections]).reshape(-1, 2)
        self._track(detections, dt)
        return self.people

    @staticmethod
    def _as_array(points) -> np.ndarray:
        if getattr(points, "dtype", None) is not None and points.dtype.names:
            if len(points) == 0:
                return np.empty((0, 4))
            return np.column_stack([
                points["x"], points["y"], points["velocity"], points["snr"],
            ]).astype(float)
        data = np.asarray(points, dtype=float)
        if data.size == 0:
            return np.empty((0, 4))
        if data.ndim != 2 or data.shape[1] != 4:
            raise ValueError("expected points as (N, 4): x, y, velocity, snr")
        return data

    def _cluster(self, pooled: np.ndarray):
        detections = []
        if len(pooled) < self.min_points:
            return detections
        labels = dbscan(pooled[:, :2], self.eps, self.min_points)
        for label in range(labels.max() + 1):
            members = pooled[labels == label]
            weights = np.maximum(members[:, 3], 1.0)
            centre = np.average(members[:, :2], axis=0, weights=weights)
            moving = members[members[:, 2] != 0, 2]
            radial = float(moving.mean()) if len(moving) else 0.0
            detections.append((centre, len(members), radial))
        return detections

    # ------------------------------------------------------------------ #

    def _track(self, detections, dt: float) -> None:
        if dt > 0:
            transition = np.eye(4)
            transition[0, 2] = transition[1, 3] = dt
            q = self.acceleration_sigma ** 2
            noise = np.zeros((4, 4))
            for a, b in ((0, 2), (1, 3)):
                noise[a, a] = q * dt ** 4 / 4
                noise[a, b] = noise[b, a] = q * dt ** 3 / 2
                noise[b, b] = q * dt ** 2
            for track in self.tracks:
                if track.coasting:
                    # Without measurements keep the position and let the
                    # velocity die out instead of extrapolating out of the room.
                    track.state[2:] *= 0.5 ** (dt / 0.5)
                    track.covariance[:2, :2] += np.eye(2) * q * dt ** 2 * 0.05
                else:
                    track.state = transition @ track.state
                    track.covariance = transition @ track.covariance @ transition.T + noise
                track.age += dt
                track.since_seen += dt

        unused = set(range(len(detections)))
        pairs = []
        for t_index, track in enumerate(self.tracks):
            for d_index, (centre, _count, _radial) in enumerate(detections):
                distance = float(np.hypot(*(centre - track.state[:2])))
                if distance <= self.gate:
                    # Prefer established tracks when two compete for one cluster.
                    pairs.append((distance - 0.2 * track.confirmed, t_index, d_index))
        used_tracks = set()
        observe = np.zeros((2, 4))
        observe[0, 0] = observe[1, 1] = 1.0
        r = np.eye(2) * self.measurement_sigma ** 2
        for _cost, t_index, d_index in sorted(pairs):
            if t_index in used_tracks or d_index not in unused:
                continue
            used_tracks.add(t_index)
            unused.discard(d_index)
            track = self.tracks[t_index]
            centre, count, radial = detections[d_index]
            innovation = centre - observe @ track.state
            s = observe @ track.covariance @ observe.T + r
            gain = track.covariance @ observe.T @ np.linalg.inv(s)
            track.state = track.state + gain @ innovation
            track.covariance = (np.eye(4) - gain @ observe) @ track.covariance
            track.hits += 1
            track.since_seen = 0.0
            track.num_points = count
            track.radial_velocity = radial
            if track.hits >= self.confirm_hits:
                track.confirmed = True

        for d_index in sorted(unused):
            centre, count, radial = detections[d_index]
            state = np.array([centre[0], centre[1], 0.0, 0.0])
            covariance = np.diag([self.measurement_sigma ** 2] * 2 + [1.0, 1.0])
            self.tracks.append(Track(self._next_id, state, covariance,
                                     num_points=count, radial_velocity=radial))
            self._next_id += 1

        survivors = []
        for track in self.tracks:
            limit = self.hold if track.confirmed else self.tentative_timeout
            in_view = (track.y > 0
                       and track.range <= self.max_range + self.gate)
            if track.since_seen <= limit and in_view:
                survivors.append(track)
        # Two tracks on one person: keep the older one.
        survivors.sort(key=lambda track: (-track.confirmed, -track.age))
        kept: List[Track] = []
        for track in survivors:
            if any(np.hypot(track.x - other.x, track.y - other.y) < self.eps / 2
                   for other in kept):
                continue
            kept.append(track)
        kept.sort(key=lambda track: track.id)
        for track in kept:
            if track.confirmed:
                track.trail.append((track.x, track.y))
        self.tracks = kept
