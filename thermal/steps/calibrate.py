"""Step 2: heat-sink resistance r_convec per bond-layer conductivity k on the 2 x 2 floorplan -> out/calibrate.json.

For each k, r_convec is fitted so that the hottest DRAM block (block centre) rises by the measured HBM temperature per
board W (anchor GEMM sweep) when the board power goes from 80 to 480 W, with the measured memory share of each board W
spread over the DRAM dies and the rest on the GPU die. Background: the DRAMPower decode map (data/inputs). Readout:
hottest 2 x 2 DRAM block, block centre (HotSpot grid_map_mode center; steps 3 and 4 read the hottest grid cell). Secant
extrapolation from r = 0.015 / 0.025 (the target is not bracketed), checked at the result within 0.2 %. At the fit: uniform PIM of 40 W per stack.
k = 1 (default) is the model used downstream; --all-k adds k = 1.5 ... 16.

    python -m thermal.steps.calibrate [--all-k]
"""
import json
import re
import sys
import time

from .. import paths
from ..model import geom as G
from ..model import hotspot as HS
from ..model import measured as M
from ..model.power import cal_row

KS_ALL = (1.0, 1.5, 2.0, 4.0, 5.5, 8.0, 16.0)
R_TRY = (0.015, 0.025)
G_LO, G_HI = 80.0, 480.0
PIM_W = 40.0


def name(k, r, kind):
    return f"cal_k{k:g}_r{r:.5f}_{kind}"


def read(nm):
    T = HS.steady(nm)
    banks = {kk: v for kk, v in T.items() if re.search(r"_B_\d+$", kk)}
    hot = max(banks, key=banks.get)
    return dict(dram_peak=banks[hot], dram_peak_block=hot, dram_mean=sum(banks.values()) / len(banks), gpu_die=T["layer_2_C_0"])


def jobs(k, r, mem_share, dram_map, pim=False):
    g = G.variant(k, r, 64, "2x2")
    cols = G.columns(g)
    dp = G_HI - G_LO
    q4 = lambda x: float(f"{x:.4f}")          # noqa: E731  (inputs of the original runs were passed with 4 decimals)
    out = [(name(k, r, "g80"), g, [cal_row(cols, q4(G_LO), dram_map)], False),
           (name(k, r, f"g480_m{mem_share:.4f}"), g, [cal_row(cols, q4(G_LO + dp * (1 - mem_share)), dram_map, q4(dp * mem_share / 5))], False)]
    if pim:
        out.append((name(k, r, "g80_pim40"), g, [cal_row(cols, q4(G_LO), dram_map, q4(PIM_W))], False))
    return out


def main(ks=(1.0,)):
    t0 = time.time()
    an = M.anchor_slopes()
    target, ms = an["hbm_per_board_W"], an["mem_per_board_W"]
    dram_map = json.load(open(paths.inputs("dram_map_decode.json")))["per_block_W"]
    hs = 0.0
    J = [j for k in ks for r in R_TRY for j in jobs(k, r, ms, dram_map)]
    hs += HS.run_all(J, label="calibrate")
    slope = lambda k, r: (read(name(k, r, f"g480_m{ms:.4f}"))["dram_peak"] - read(name(k, r, "g80"))["dram_peak"]) / (G_HI - G_LO)  # noqa: E731
    fits = {}
    for k in ks:
        sl = [slope(k, r) for r in R_TRY]
        fits[k] = (R_TRY[0] + (target - sl[0]) * (R_TRY[1] - R_TRY[0]) / (sl[1] - sl[0]), sl)
    for _ in range(2):                         # the hottest block can switch with r: secant steps from r*
        hs += HS.run_all([j for k, (rs, _) in fits.items() for j in jobs(k, round(rs, 5), ms, dram_map)], label="calibrate")
        for k, (rs, sl) in list(fits.items()):
            r1 = round(rs, 5)
            s1 = slope(k, r1)
            r0 = min(R_TRY, key=lambda r: abs(r - r1))
            s0 = slope(k, r0)
            if abs(s1 / target - 1) < 0.002:
                fits[k] = (r1, sl)
                continue
            fits[k] = (r1 + (target - s1) * (r0 - r1) / (s0 - s1), sl)
    fits = {k: (round(rs, 5), sl) for k, (rs, sl) in fits.items()}
    hs += HS.run_all([j for k, (rs, _) in fits.items() for j in jobs(k, rs, ms, dram_map, pim=True)], label="calibrate")
    out = dict(target_hbm_per_board_W=target, mem_per_board_W=ms, board_W=[G_LO, G_HI], pim_W_per_stack=PIM_W, k={})
    for k, (rs, sl) in fits.items():
        b, s480, p = read(name(k, rs, "g80")), read(name(k, rs, f"g480_m{ms:.4f}")), read(name(k, rs, "g80_pim40"))
        out["k"][f"k{k:g}"] = dict(
            k=k, r_convec=rs, slope_at_r_try=sl, slope_check=(s480["dram_peak"] - b["dram_peak"]) / (G_HI - G_LO),
            gpu_per_board_w=(s480["gpu_die"] - b["gpu_die"]) / (G_HI - G_LO),
            pim_peak_per_w_per_stack=(p["dram_peak"] - b["dram_peak"]) / PIM_W,
            pim_mean_per_w_per_stack=(p["dram_mean"] - b["dram_mean"]) / PIM_W,
            pim_to_gpu_per_w_per_stack=(p["gpu_die"] - b["gpu_die"]) / PIM_W,
            hot_block_base=b["dram_peak_block"], hot_block_pim40=p["dram_peak_block"])
    out["seconds"] = round(time.time() - t0, 1)
    out["hotspot_seconds"] = round(hs, 1)
    path = paths.out("calibrate.json")
    try:                                       # keep other k from an earlier --all-k run
        old = json.load(open(path))["k"]
        out["k"] = {**{kk: v for kk, v in old.items() if kk not in out["k"]}, **out["k"]}
    except (OSError, ValueError, KeyError):
        pass
    json.dump(out, open(path, "w"), indent=1)
    for kk, v in out["k"].items():
        print(f"[calibrate] {kk}: r_convec {v['r_convec']:.5f}, slope {v['slope_check']:.5f} (target {target:.5f}), "
              f"uniform PIM {v['pim_peak_per_w_per_stack']:.3f} K per W per stack", flush=True)
    return out


def r_convec(k=1.0):
    """Calibrated r_convec at bond-layer conductivity k (needs this step)."""
    try:
        return json.load(open(paths.out("calibrate.json")))["k"][f"k{k:g}"]["r_convec"]
    except (OSError, KeyError) as e:
        raise SystemExit(f"no calibrated r_convec for k = {k:g} in {paths.out('calibrate.json')}: run the calibrate step") from e


if __name__ == "__main__":
    main(KS_ALL if "--all-k" in sys.argv else (1.0,))
