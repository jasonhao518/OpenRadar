"""Fixed radar reflection evidence map; unseen cells remain unknown.

Input is the unshifted TI range-azimuth power matrix. This is a horizontal
projection, not an elevation-resolved floor plan or calibrated occupancy.
"""
import numpy as np


class ReflectionMap:
    def __init__(self, limit=5.0, max_range=10.0, cell_size=0.1,
                 range_step=0.1953125, fps=4.0, history=30.0, threshold_db=6.0):
        if min(limit, max_range, cell_size, range_step, fps, history) <= 0:
            raise ValueError("map dimensions, frame rate and history must be positive")
        self.x = np.arange(-limit + cell_size / 2, limit, cell_size)
        self.y = np.arange(cell_size / 2, max_range, cell_size)
        xx, yy = np.meshgrid(self.x, self.y)
        self.ranges = np.hypot(xx, yy)
        self.sines = xx / np.maximum(self.ranges, 1e-9)
        self.range_step = range_step
        self.fps = fps
        self.history = history
        self.threshold_db = threshold_db
        self.evidence = np.zeros_like(xx)
        self.current = np.zeros_like(xx, dtype=bool)
        self.last_frame = None

    def clear(self):
        self.evidence.fill(0)
        self.current.fill(False)
        self.last_frame = None

    def update(self, heatmap, frame_number, min_range=0.25):
        power = np.fft.fftshift(np.asarray(heatmap, dtype=float), axes=1)
        if power.ndim != 2 or min(power.shape) < 3:
            raise ValueError("expected a range-by-azimuth power matrix")
        nr, na = power.shape
        # Per-angle median estimates background; require a local range peak
        # to avoid filling space between a reflector and the radar.
        valid_rows = np.arange(nr) * self.range_step >= min_range
        if not np.any(valid_rows):
            raise ValueError("no range bins inside map bounds")
        noise = np.maximum(np.median(power[valid_rows], axis=0), 1.0)
        contrast = 10 * np.log10(np.maximum(power, 1.0) / noise)
        peaks = np.zeros_like(power, dtype=bool)
        peaks[1:-1] = (power[1:-1] > power[:-2]) & (power[1:-1] >= power[2:])
        peaks &= contrast >= self.threshold_db
        ri = np.rint(self.ranges / self.range_step).astype(int)
        ai = np.rint(self.sines * na / 2 + na // 2).astype(int)
        valid = ((ri < nr) & (ai >= 0) & (ai < na)
                 & (self.ranges >= min_range) & (np.abs(self.sines) <= np.sin(np.radians(60))))
        self.current = valid & peaks[np.clip(ri, 0, nr-1), np.clip(ai, 0, na-1)]
        delta = 1 if self.last_frame is None else max(1, int(frame_number)-self.last_frame)
        decay = np.exp(-min(delta / self.fps, self.history) / self.history)
        self.evidence *= decay
        self.evidence += (1-decay) * self.current
        self.last_frame = int(frame_number)
        return self.evidence
