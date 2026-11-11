"""Energy balance of HotSpot steady runs: power in (ptrace or known) vs heat out through the convection boundary.

Heat out (HotSpot temperature_grid.c single_iteration_steady_grid / _pack, package.c populate_package_R):
  sink core cells: (T - Tamb) / rz,  rz = t_sink / (k_sink cw ch) + r_convec s_sink^2 / (cw ch)
  inner sink periphery N/S: (T - Tamb) / (r_hs_c_per_y + r_amb_c_per_y), W/E: (.. _x)
  outer sink periphery: (T - Tamb) / (r_hs_per + r_amb_per)
"""
import os
import re

import numpy as np

W, H = 0.04553, 0.03657


def cfg(gdir):
    c = {}
    for ln in open(os.path.join(gdir, "run.config")):
        m = re.match(r"\s*-(\w+)\s+(\S+)", ln)
        if m:
            try:
                c[m.group(1)] = float(m.group(2))
            except ValueError:
                pass
    return c


def heat_out(sink_K, inode, c, grid):
    """sink_K: grid x grid sink-layer temperatures (K); inode: 12 package node temperatures (K)."""
    amb = c["ambient"]
    cw, ch = W / grid, H / grid
    ks, ts, ss, sp = c["k_sink"], c["t_sink"], c["s_sink"], c["s_spreader"]
    rc = c["r_convec"]
    rz = ts / (ks * cw * ch) + rc * ss * ss / (cw * ch)
    core = ((sink_K - amb) / rz).sum()
    getr = lambda k, l, a: l / (k * a)  # noqa: E731
    a_x = (sp + H) * (sp - W) / 4.0
    a_y = (sp + W) * (sp - H) / 4.0
    a_o = (ss * ss - sp * sp) / 4.0
    r_cx = getr(ks, ts, a_x) + rc * ss * ss / a_x
    r_cy = getr(ks, ts, a_y) + rc * ss * ss / a_y
    r_o = getr(ks, ts, a_o) + rc * ss * ss / a_o
    inner = (inode[4] - amb) / r_cx + (inode[5] - amb) / r_cx + (inode[6] - amb) / r_cy + (inode[7] - amb) / r_cy
    outer = sum((inode[i] - amb) / r_o for i in (8, 9, 10, 11))
    return dict(core=core, inner=inner, outer=outer, total=core + inner + outer)


def read_grid(p, grid):
    return np.array([float(ln.split()[1]) for ln in open(p) if ln.strip()]).reshape(grid, grid)


def read_inodes(steady):
    d = {}
    for ln in open(steady):
        k, v = ln.split()
        if k.startswith("inode_"):
            d[int(k[6:])] = float(v)
    return [d[i] for i in range(12)]


def ptrace_total(pt):
    lines = [ln for ln in open(pt).read().splitlines() if ln.strip()]
    return sum(float(x) for x in lines[1].split())
