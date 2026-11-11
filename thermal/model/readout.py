"""Temperature readouts of HotSpot grid fields (row 0 of a layer dump is the top edge of the package).

cell     hottest grid cell of the eight DRAM dies inside the five stack footprints (the HBM temperature of the model)
coarse   hottest centre of the original 2 x 2 DRAM blocks (HotSpot grid_map_mode center)
block    hottest centre of the bank-floorplan blocks
bottom   mean of the bottom DRAM die over a stack footprint;  base: mean of the base die over a stack footprint
gpu      grid cells of the GPU die
"""
import math

import numpy as np

from . import geom as G


def _tc(v):
    r = math.floor(v + 0.5)
    return int(r) if abs(v - r) < 1e-6 else int(math.ceil(v))


def _tf(v):
    r = math.floor(v + 0.5)
    return int(r) if abs(v - r) < 1e-6 else int(math.floor(v))


def centre(L, x, y, w, h, n):
    """HotSpot grid_map_mode center readout of block (x, y, w, h) from a layer grid L (temperature_grid.c)."""
    cw, ch = G.W / n, G.H / n
    i1, i2 = n - _tc((y + h) / ch), n - _tf(y / ch)
    j1, j2 = _tf(x / cw), _tc((x + w) / cw)
    ci1, cj1 = (i1 + i2) // 2, (j1 + j2) // 2
    ci2, cj2 = ci1 - (1 if (i2 - i1) % 2 == 0 else 0), cj1 - (1 if (j2 - j1) % 2 == 0 else 0)
    return (L[ci1, cj1] + L[ci2, cj1] + L[ci1, cj2] + L[ci2, cj2]) / 4


def coarse_points(fp):
    """The original 2 x 2 blocks of every stack: (x, y, w, h)."""
    pts = []
    for s, (x0, y0, x1, y1) in fp.items():
        w, h = (x1 - x0) / 2, (y1 - y0) / 2
        for i in range(2):
            for j in range(2):
                pts.append((x0 + i * w, y0 + j * h, w, h))
    return pts


def cell_centres(n):
    xc = (np.arange(n) + 0.5) * G.W / n
    yc = G.H - (np.arange(n) + 0.5) * G.H / n
    return np.meshgrid(xc, yc)


class Readout:
    """Index sets of one grid size. vec(F) gives linear readout vectors, so a readout of a superposition is the same
    superposition of the vectors; the maximum (cell, coarse, gpu) or hottest-stack value is taken afterwards."""

    def __init__(self, grid):
        fp, _ = G.footprints()
        n = grid
        X, Y = cell_centres(n)
        inside = np.zeros((n, n), bool)
        for (x0, y0, x1, y1) in fp.values():
            inside |= (X > x0) & (X < x1) & (Y > y0) & (Y < y1)
        self.inside = inside
        self.gpu = (X > G.GX0) & (X < G.GX1) & (Y > G.GY0) & (Y < G.GY1)
        self.grid = n
        self.fp = fp
        self.coarse_pts = coarse_points(fp)
        self.stack_masks = [((X > x0) & (X < x1) & (Y > y0) & (Y < y1)) for (x0, y0, x1, y1) in fp.values()]

    def vec(self, F):
        cells = np.concatenate([F[f"L{4 + 2 * d}"][self.inside] for d in range(8)])
        coarse = np.array([centre(F[f"L{4 + 2 * d}"], *r, self.grid) for d in range(8) for r in self.coarse_pts])
        bottom = np.array([F["L4"][mk].mean() for mk in self.stack_masks])
        basem = np.array([F["L2"][mk].mean() for mk in self.stack_masks])
        return dict(cell=cells, coarse=coarse, gpu=F["L2"][self.gpu], bottom=bottom, base=basem)

    def hottest(self, F, m=None):
        """Hottest-DRAM readouts of one field set: cell, coarse, gpu, and block (bank-floorplan block centres)."""
        out = dict(cell=max(F[f"L{4 + 2 * d}"][self.inside].max() for d in range(8)),
                   coarse=max(centre(F[f"L{4 + 2 * d}"], *r, self.grid) for d in range(8) for r in self.coarse_pts),
                   gpu=F["L2"][self.gpu].max())
        if m is not None:
            out["block"] = max(centre(F[f"L{b['layer']}"], b["x"], b["y"], b["w"], b["h"], self.grid) for b in m["blocks"])
        return out


_RO = {}


def readout(grid):
    if grid not in _RO:
        _RO[grid] = Readout(grid)
    return _RO[grid]


def combine(parts):
    """Superpose field sets: parts = [(weight, fields)]."""
    out = None
    for w, F in parts:
        if out is None:
            out = {k: w * v for k, v in F.items()}
        else:
            for k, v in F.items():
                out[k] = out[k] + w * v
    return out
