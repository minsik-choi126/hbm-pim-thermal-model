"""Package geometry: the H100 5-stack floorplans of thermal/geometry/h100_5stack turned into HotSpot run directories.

A run directory (variant) fixes four things:
  k      bond-layer conductivity inside the HBM stack footprints (W/m-K; the layers between the base die and the DRAM
         dies and between DRAM dies take resistivity 1 / k there, 0.25 m-K/W elsewhere)
  r      heat-sink convection resistance r_convec (K/W)
  grid   HotSpot grid rows = cols
  fp     DRAM-die floorplan: "2x2" (four blocks per die) or a bank floorplan "a50_par", "a75_par", "a50_perp", "a75_perp"

Common to all variants: the silicon column above the GPU die (the filler over the GPU in every DRAM-die layer is
silicon), 0.1 ms transient rows.

Bank floorplan (each stack footprint, 8.00 mm across x 12.19 mm along the GPU edge, all 8 DRAM dies): a DRAM die holds
the 128 banks of two HBM3 channels (64 each, JESD238B.01 Table 4), drawn as 64 two-bank cells. A central TSV stripe
splits the die into two halves, one channel each (32 cells as 4 across the stripe x 8 along it); a quadrant of 16 cells
(32 banks) is one pseudo channel. Each cell = bank array + column-decoder strip on the stripe side (where the MAC unit
sits) + a row-decoder strip. The "bank" index in the code numbers the 64 cells. a = bank-array share of the die: 0.50 (stripe 20 % of the die) or 0.75
(stripe 12.5 %). "par": stripe parallel to the GPU-facing edge (halves = GPU side and far side); "perp": across it.
"""
import json
import math
import os
import re
import shutil

from .. import paths

STACKS = 5
AMB = 50.0                                   # ambient, degC (config: 323.15 K)
DT_MS = 0.1                                  # transient row length
W, H = 0.04553, 0.03657                      # package box (m)
GX0, GX1, GY0, GY1 = 0.0085, 0.03703, 0.00402, 0.03255     # GPU die
SI_RES, SI_CP = 0.007, 1750000.0
BOND_RES, BOND_CP = 0.25, 4000000.0
FILL_RES, FILL_CP = 0.13, 2875000.0
TIM_TILES = [  # bond layers between DRAM dies: name, x0, x1, y0, y1, kind (hbm: 1 / k, bond: 0.25, si: GPU column)
    ("TB_L", 0.0, 0.008, 0.0, H, "hbm"), ("TB_g0", 0.008, GX0, 0.0, H, "bond"),
    ("TB_c0", GX0, GX1, 0.0, GY0, "bond"), ("TB_GPU", GX0, GX1, GY0, GY1, "si"), ("TB_c1", GX0, GX1, GY1, H, "bond"),
    ("TB_g1", GX1, 0.03753, 0.0, H, "bond"), ("TB_R", 0.03753, W, 0.0, 0.02438, "hbm"), ("TB_E", 0.03753, W, 0.02438, H, "bond"),
]
HBM_BOND_LAYERS = {3, 5, 7, 9, 11, 13, 15, 17}
LAYERS = [2] + [4 + 2 * d for d in range(8)]          # base dies + GPU die, DRAM dies 1..8
AREA = {0.75: 0.125, 0.50: 0.20}                      # bank-array share a -> TSV stripe share c
FLOORPLANS = {"a50_par": (0.50, "par"), "a75_par": (0.75, "par"), "a50_perp": (0.50, "perp"), "a75_perp": (0.75, "perp")}
BANK_RE = re.compile(r"^B_(\d+)$")
LC_RE = re.compile(r"^LC_(\d+)_(\d+)$")


def tag(k, r, grid, fp):
    return f"{fp}_k{k:g}_r{r:.5f}_g{grid}"


def footprints():
    """Stack footprints {stack: (x0, y0, x1, y1)} from the 2 x 2 DRAM blocks, and the GPU-die centre x."""
    fp = {}
    for ln in open(os.path.join(paths.GEOM_SRC, "mem_bank_1.flp")):
        p = ln.split()
        if p and BANK_RE.match(p[0]):
            s = int(p[0][2:]) // 32
            w, h, x, y = map(float, p[1:5])
            a = fp.get(s, (x, y, x + w, y + h))
            fp[s] = (min(a[0], x), min(a[1], y), max(a[2], x + w), max(a[3], y + h))
    gx = None
    for ln in open(os.path.join(paths.GEOM_SRC, "core_and_mem_ctrl.flp")):
        p = ln.split()
        if p and p[0] == "C_0":
            gx = float(p[3]) + float(p[1]) / 2
    assert len(fp) == STACKS
    return fp, gx


def die_blocks(a, orient, fp, gx):
    """Blocks of one DRAM die for every stack: dicts (kind, x, y, w, h, stack, half, pch, bank, near, lower).
    kind: ar bank array, cd column-decoder strip, rd row-decoder strip, ts TSV stripe."""
    c = AREA[a]
    out = []
    for s, (x0, y0, x1, y1) in fp.items():
        Wd, Hd = x1 - x0, y1 - y0
        gpu_hi = (x0 + x1) / 2 < gx                 # GPU-facing edge at x1 (left stacks) or x0 (right stacks)
        Lp, Lq = (Wd, Hd) if orient == "par" else (Hd, Wd)     # (p, q): p across the stripe, q along it
        wc = c * Lp
        bp, bq = (Lp - wc) / 2 / 4, Lq / 8
        kk = 1 - math.sqrt(a / (1 - c))             # both decoder strips take the same fraction of the bank cell
        wcd, hrd = kk * bp, kk * bq

        def put(kind, p0, q0, dp, dq, half, pch, bank):
            if orient == "par":
                x, y, w, h = x0 + p0, y0 + q0, dp, dq
            else:
                x, y, w, h = x0 + q0, y0 + p0, dq, dp
            out.append(dict(kind=kind, x=x, y=y, w=w, h=h, stack=s, half=half, pch=pch, bank=bank))
        for half in (0, 1):
            for iq in range(8):
                pch = half * 2 + iq // 4
                q0 = iq * bq
                put("rd", 0.0 if half == 0 else Lp / 2 + wc / 2, q0, (Lp - wc) / 2, hrd, half, pch, -1)
                for ip in range(4):                  # ip = 0 next to the stripe
                    bank = half * 32 + iq * 4 + ip
                    if half == 0:
                        cell0 = Lp / 2 - wc / 2 - (ip + 1) * bp
                        cd0, arr0 = cell0 + bp - wcd, cell0
                    else:
                        cell0 = Lp / 2 + wc / 2 + ip * bp
                        cd0, arr0 = cell0, cell0 + wcd
                    put("cd", cd0, q0 + hrd, wcd, bq - hrd, half, pch, bank)
                    put("ar", arr0, q0 + hrd, bp - wcd, bq - hrd, half, pch, bank)
                p0 = Lp / 2 - wc / 2 if half == 0 else Lp / 2
                put("ts", p0, q0, wc / 2, bq, half, pch, -1)
        for b in out:
            if b["stack"] == s:
                cx = b["x"] + b["w"] / 2
                b["near"] = (cx > (x0 + x1) / 2) == gpu_hi
                b["lower"] = b["y"] + b["h"] / 2 < (y0 + y1) / 2
    return out


def qfs_sel(b):
    """Quarter of each DRAM die: the far-side lower 2 x 2 block of the original floorplan."""
    return (not b["near"]) and b["lower"]


def _with_column(src_lines):
    out = []
    for ln in src_lines:
        if ln.split() and ln.split()[0] == "X_c":
            out += [f"X_c0\t{GX1 - GX0:.6f}\t{GY0:.6f}\t{GX0:.6f}\t0.000000\t{FILL_CP:.0f}\t{FILL_RES}",
                    f"X_cS\t{GX1 - GX0:.6f}\t{GY1 - GY0:.6f}\t{GX0:.6f}\t{GY0:.6f}\t{SI_CP:.0f}\t{SI_RES}",
                    f"X_c1\t{GX1 - GX0:.6f}\t{H - GY1:.6f}\t{GX0:.6f}\t{GY1:.6f}\t{FILL_CP:.0f}\t{FILL_RES}"]
        else:
            out.append(ln)
    assert sum(1 for ln in out if ln.startswith("X_cS")) == 1
    return out


def variant(k, r, grid=64, fp="2x2"):
    """Build (once) and return the HotSpot run directory of (k, r, grid, floorplan)."""
    dst = os.path.join(paths.RUNS, "geom", tag(k, r, grid, fp))
    if os.path.isdir(dst):
        return dst
    tmp = dst + ".tmp"
    shutil.rmtree(tmp, ignore_errors=True)
    os.makedirs(tmp)
    src = paths.GEOM_SRC
    for f in os.listdir(src):
        if f.endswith(".flp"):
            shutil.copy(os.path.join(src, f), tmp)
    for i in range(1, 9):
        p = os.path.join(tmp, f"mem_bank_{i}.flp")
        lines = _with_column(open(p).read().splitlines())
        open(p, "w").write("\n".join(lines) + "\n")
    res = {"hbm": (BOND_CP, 1.0 / k), "bond": (BOND_CP, BOND_RES), "si": (SI_CP, SI_RES)}
    with open(os.path.join(tmp, "tim_hbm.flp"), "w") as f:
        for name, x0, x1, y0, y1, kind in TIM_TILES:
            cp, rr = res[kind]
            f.write(f"{name}\t{x1 - x0:.6f}\t{y1 - y0:.6f}\t{x0:.6f}\t{y0:.6f}\t{cp:.0f}\t{rr:.6g}\n")
    blocks = []
    for blk in open(os.path.join(src, "stack.lcf")).read().split("\n\n"):
        data = [ln for ln in blk.strip().split("\n") if ln and not ln.startswith("#")]
        if len(data) == 7 and data[6] == "tim.flp" and int(data[0]) in HBM_BOND_LAYERS:
            blk = blk.replace("tim.flp", "tim_hbm.flp")
        blocks.append(blk)
    lcf = "\n\n".join(blocks)
    assert lcf.count("tim_hbm.flp") == len(HBM_BOND_LAYERS)
    open(os.path.join(tmp, "stack.lcf"), "w").write(lcf)
    conf = open(paths.CONFIG).read()
    for key, val in (("r_convec", f"{r:.8f}"), ("grid_rows", str(grid)), ("grid_cols", str(grid)),
                     ("sampling_intvl", f"{DT_MS * 1e-3:g}")):
        conf, n = re.subn(r"(-%s\s+)\S+" % key, lambda m: m.group(1) + val, conf, count=1)
        assert n == 1, key
    open(os.path.join(tmp, "run.config"), "w").write(conf)
    if fp != "2x2":
        a, orient = FLOORPLANS[fp]
        fps, gx = footprints()
        bl = die_blocks(a, orient, fps, gx)
        for s, (x0, y0, x1, y1) in fps.items():          # tiling check per stack
            ar = sum(b["w"] * b["h"] for b in bl if b["stack"] == s)
            assert abs(ar / ((x1 - x0) * (y1 - y0)) - 1) < 1e-9
        meta = []
        for d in range(8):
            p = os.path.join(tmp, f"mem_bank_{d + 1}.flp")
            lines = open(p).read().splitlines()
            keep = [ln for ln in lines if not (ln.split() and BANK_RE.match(ln.split()[0]))]
            new = []
            for i, b in enumerate(bl):                  # short names: HotSpot reads at most 65536 characters per line
                nm = f"{b['kind'][0]}{d}{i:03d}"
                new.append(f"{nm}\t{b['w']:.9f}\t{b['h']:.9f}\t{b['x']:.9f}\t{b['y']:.9f}")
                meta.append(dict(b, name=nm, die=d, layer=4 + 2 * d))
            open(p, "w").write("\n".join([keep[0]] + new + keep[1:]) + "\n")
        json.dump(dict(a=a, c=AREA[a], orient=orient, stacks=fps, gx=gx, blocks=meta), open(os.path.join(tmp, "blocks.json"), "w"))
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    os.rename(tmp, dst)
    return dst


def load_blocks(gdir):
    """Bank-floorplan metadata of a variant: stacks, gx, blocks (with die and layer), info (name -> block)."""
    m = json.load(open(os.path.join(gdir, "blocks.json")))
    m["stacks"] = {int(s): v for s, v in m["stacks"].items()}
    m["info"] = {b["name"]: b for b in m["blocks"]}
    m["_dir"] = gdir
    return m


def columns(gdir):
    """ptrace columns: the blocks of every powered layer, in layer order."""
    recs = [s.strip() for s in open(os.path.join(gdir, "stack.lcf")) if s.strip() and not s.strip().startswith("#")]
    assert recs and len(recs) % 7 == 0
    cols = []
    for i in range(0, len(recs), 7):
        if recs[i + 2].upper() == "Y":
            f = os.path.join(gdir, os.path.basename(recs[i + 6]))
            cols += [s.split()[0] for s in (ln.strip() for ln in open(f)) if s and not s.startswith("#")]
    return cols


def base_die_blocks(gdir):
    """Base-die blocks per stack ({stack: [LC_s_i]}) and the two of each stack on the GPU side."""
    xs = {}
    for ln in open(os.path.join(gdir, "core_and_mem_ctrl.flp")):
        p = ln.split()
        if p and LC_RE.match(p[0]):
            xs[p[0]] = float(p[3]) + float(p[1]) / 2
    _, gx = footprints()
    allb, phy = {}, {}
    for c in xs:
        allb.setdefault(int(LC_RE.match(c).group(1)), []).append(c)
    for s, names in allb.items():
        phy[s] = sorted(names, key=lambda c: abs(xs[c] - gx))[:2]
    assert len(allb) == STACKS and all(len(v) == 4 for v in allb.values())
    return allb, phy
