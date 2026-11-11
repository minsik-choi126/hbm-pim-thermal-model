"""Power maps (ptrace rows). The HotSpot model is linear (leakage off, fixed r_convec), so every steady run is a unit
response of one source and any power map is a superposition of them.

Unit sources (W per stack in each of the five stacks, except gpu):
  gpu       GPU die C_0, 1000 W
  uni       DRAM dies, uniform over the die area of all eight dies (memory power spread like the DRAM map)
  base      base die, uniform over its four blocks
  basephy   the two base-die blocks on the GPU side (PHY half)
  array     bank arrays of every bank of every die
  edge      column-decoder strips of every bank (MAC units)
  stripe    TSV stripe (selector only; no step runs it)
  qfs       one quarter of each DRAM die (far side, lower half): one 2 x 2 block of the original floorplan
  half_far, half_dies, quarter_far, quarter_dies
            PIM in some channels only: bank arrays with the MAC share at the bank edge (see SUBSETS)
"""
from . import geom as G

GPU_W, UNIT_W = 1000.0, 200.0        # unit amplitudes (well above HotSpot's 0.01 K output rounding)
# MAC share of PIM energy in AttAcc's table energy (0.16 / 4.56 pJ/B = 3.5 %). At the paper's in-bank energies the share
# is 0.6-1.1 %, covered by the 0.9 % case of steps/bank.py (f_bank +0.3 %).
E_TABLE = 0.16 / 4.56
SUBSETS = {   # name: (description, bank selector); PIM limited to some channels, same W per stack
    "half_far": ("far-side channel (32 cells, 64 banks) of every die", lambda b: not b["near"]),
    "half_dies": ("all banks of dies 1-4 only", lambda b: b["die"] < 4),
    "quarter_far": ("far-side lower pseudo channel (16 cells, 32 banks) of every die", lambda b: (not b["near"]) and b["lower"]),
    "quarter_dies": ("all banks of dies 1-2 only", lambda b: b["die"] < 2),
}
SRC_CORE = ("gpu", "uni", "base", "basephy", "array", "edge")
SRC_SUB = tuple(SUBSETS)
FP_FREE = ("gpu", "uni", "base", "basephy")     # do not depend on the bank floorplan (run on a50_par)


def fmt(x):
    return f"{x:.6f}".rstrip("0").rstrip(".") or "0"


def spread(v, blocks, W, sel):
    """Add W per stack, area weighted over the blocks sel picks, in every stack."""
    for s in range(G.STACKS):
        tg = [b for b in blocks if b["stack"] == s and sel(b)]
        assert tg, "empty selection"
        A = sum(b["w"] * b["h"] for b in tg)
        for b in tg:
            v[b["name"]] += W * b["w"] * b["h"] / A


def bank_maps(W, e, sub=lambda b: True):
    """PIM power W per stack: (1 - e) in the bank arrays and e in the column-decoder strips of the banks sub picks."""
    out = [((1 - e) * W, lambda b: b["kind"] == "ar" and sub(b))]
    if e:
        out.append((e * W, lambda b: b["kind"] == "cd" and sub(b)))
    return out


SEL = {"uni": lambda b: True, "array": lambda b: b["kind"] == "ar", "edge": lambda b: b["kind"] == "cd",
       "stripe": lambda b: b["kind"] == "ts", "qfs": G.qfs_sel}


def amplitude(src):
    return GPU_W if src == "gpu" else UNIT_W


def source_row(src, cols, m, amp=None):
    """ptrace row of one unit source on a bank-floorplan variant (m = geom.load_blocks)."""
    amp = amplitude(src) if amp is None else amp
    v = dict.fromkeys(cols, 0.0)
    if src == "gpu":
        v["C_0"] = amp
    elif src in ("base", "basephy"):
        allb, phy = G.base_die_blocks(m["_dir"])
        for s in range(G.STACKS):
            tg = allb[s] if src == "base" else phy[s]
            for c in tg:
                v[c] += amp / len(tg)
    elif src in SEL:
        spread(v, m["blocks"], amp, SEL[src])
    else:
        for w, sel in bank_maps(amp, E_TABLE, SUBSETS[src][1]):
            spread(v, m["blocks"], w, sel)
    return [v[c] for c in cols]


def cal_row(cols, gpu_w, dram_map, extra_w=0.0):
    """ptrace row on a 2 x 2 variant: GPU die, the DRAM map (W per block, one stack, replicated) and extra W per stack
    spread uniformly over the DRAM blocks of every stack."""
    v = dict.fromkeys(cols, 0.0)
    v["C_0"] = gpu_w
    for c in cols:
        mt = G.BANK_RE.match(c)
        if mt:
            v[c] = dram_map[int(mt.group(1)) % 32] + (extra_w * G.STACKS / (32 * G.STACKS) if extra_w else 0.0)
    return [v[c] for c in cols]
