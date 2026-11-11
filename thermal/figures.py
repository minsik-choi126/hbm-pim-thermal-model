"""Figures of the thermal model, drawn from the geometry, the steady HotSpot fields in runs/, out/recal.json and
out/beta.json.

    python -m thermal.figures [--out DIR]          # default: <repo>/figures

thermal_package.png       package floorplan, top view
thermal_dram_die.png      one DRAM die of one stack (bank floorplan a50_par)
thermal_heatmap.png       hottest DRAM die, H100 GPU-only decode and GPU+PIM at the threshold (the operating point
                          where beta is evaluated), rise above ambient, and their difference (x = 0 model of the
                          recalibration, needs the beta step)
thermal_stack_layers.png  base die and the eight DRAM dies of the hottest stack, GPU+PIM at the threshold, rise above ambient
thermal_calibration.png   row-miss ratio and beta against the base-die share x (needs the recal step)
"""
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

from . import paths  # noqa: E402
from .model import geom as G  # noqa: E402
from .model import hotspot as HS  # noqa: E402
from .model import power as P  # noqa: E402

INK, INK2, MUTED, GRIDC, SURF = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df", "#fcfcfb"
BLUE, ORANGE, AQUA, YELLOW, GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#c9c8c3"
HEAT = LinearSegmentedColormap.from_list("heat", ["#fdf1e8", "#f6b48f", "#eb6834", "#b8461b", "#6e2408"])
DIVERGE = LinearSegmentedColormap.from_list("diverge", ["#0d366b", "#3987e5", "#9ec5f4", "#efeeeb", "#f6b48f", "#eb6834", "#6e2408"])
MM = 1e3

plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.titlesize": 10, "axes.titlecolor": INK, "figure.facecolor": "white",
                     "savefig.facecolor": "white", "axes.spines.top": False, "axes.spines.right": False})


def outlines(ax, lw=1.0, color=INK, label=False, gpu_label="GPU die"):
    """GPU-die box, the five stack boxes and the dark site (dashed)."""
    fp, _ = G.footprints()
    lab = dict(fontsize=7, color=color, bbox=dict(fc="white", ec="none", pad=0.8, alpha=0.8))
    ax.add_patch(Rectangle((G.GX0 * MM, G.GY0 * MM), (G.GX1 - G.GX0) * MM, (G.GY1 - G.GY0) * MM, fill=False, lw=lw, ec=color))
    for s, (x0, y0, x1, y1) in fp.items():
        ax.add_patch(Rectangle((x0 * MM, y0 * MM), (x1 - x0) * MM, (y1 - y0) * MM, fill=False, lw=lw, ec=color))
        if label:
            ax.text((x0 + x1) / 2 * MM, y0 * MM + 0.8, f"stack {s}", ha="center", va="bottom", **lab)
    x0, y0 = fp[3][0], fp[4][3]
    ax.add_patch(Rectangle((x0 * MM, y0 * MM), (fp[3][2] - x0) * MM, G.H * MM - y0 * MM, fill=False, lw=lw, ec=color,
                           ls=(0, (3, 2))))
    if label:
        ax.text((x0 + fp[3][2]) / 2 * MM, y0 * MM + 0.8, "dark\nsite", ha="center", va="bottom", **lab)
        ax.text((G.GX0 + G.GX1) / 2 * MM, G.GY0 * MM + 0.8, gpu_label, ha="center", va="bottom", **lab)


def package(path):
    fp, _ = G.footprints()
    fig, ax = plt.subplots(figsize=(6.4, 5.0))
    ax.add_patch(Rectangle((0, 0), G.W * MM, G.H * MM, fc="#f0efec", ec=MUTED, lw=1))
    sw, sh = (fp[0][2] - fp[0][0]) * MM, (fp[0][3] - fp[0][1]) * MM
    ax.text(G.W * MM / 2, -1.6, f"silicon interposer {G.W * MM:.2f} x {G.H * MM:.2f} mm;  stack {sw:.2f} x {sh:.2f} mm, "
            f"{(G.GX0 - fp[0][2]) * MM:.1f} mm from the GPU die", ha="center", va="top", color=INK2, fontsize=8)
    gw, gh = (G.GX1 - G.GX0) * MM, (G.GY1 - G.GY0) * MM
    ax.add_patch(Rectangle((G.GX0 * MM, G.GY0 * MM), gw, gh, fc="#cde2fb", ec=BLUE, lw=1.5))
    ax.text((G.GX0 + G.GX1) / 2 * MM, (G.GY0 + G.GY1) / 2 * MM, f"GPU die\n{gw:.2f} x {gh:.2f} mm", ha="center", va="center",
            color=INK, fontsize=10)
    for s, (x0, y0, x1, y1) in fp.items():
        ax.add_patch(Rectangle((x0 * MM, y0 * MM), (x1 - x0) * MM, (y1 - y0) * MM, fc="#fbd9c9", ec=ORANGE, lw=1.5))
        ax.text((x0 + x1) / 2 * MM, (y0 + y1) / 2 * MM, f"HBM\nstack {s}", ha="center", va="center", color=INK)
    x0, y0 = fp[3][0], fp[4][3]
    ax.add_patch(Rectangle((x0 * MM, y0 * MM), (fp[3][2] - x0) * MM, G.H * MM - y0 * MM, fc="white", ec=GRAY, lw=1.2, ls="--"))
    ax.text((x0 + fp[3][2]) / 2 * MM, (y0 * MM + G.H * MM) / 2, "dark\nsite", ha="center", va="center", color=MUTED)
    ax.text(G.W * MM + 1.8, G.H * MM / 2,
            "layers, bottom to top:\ninterposer\nbond\nGPU die + 5 base dies\nbond\n8 x (DRAM die, bond)\nheat spreader, heat sink",
            va="center", ha="left", fontsize=7.5, color=INK2, linespacing=1.4, bbox=dict(fc="white", ec=GRIDC, pad=4))
    ax.set_xlim(-0.5, G.W * MM + 0.5)
    ax.set_ylim(-4.5, G.H * MM + 0.5)
    ax.set_aspect("equal")
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    ax.set_title("H100 package model, top view (HotSpot 6.0 grid model)", loc="left")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def dram_die(path, fp_name="a50_par"):
    a, orient = G.FLOORPLANS[fp_name]
    fps, gx = G.footprints()
    bl = [b for b in G.die_blocks(a, orient, fps, gx) if b["stack"] == 0]
    x0, y0, x1, y1 = fps[0]
    col = {"ar": ("bank array (one cell = two banks)", "#cde2fb", BLUE), "cd": ("column decoder (MAC unit)", ORANGE, "#b8461b"),
           "rd": ("row decoder", "#bfe8d8", AQUA), "ts": ("TSV stripe", "#e4e3df", MUTED)}
    fig, ax = plt.subplots(figsize=(4.6, 5.6))
    for b in bl:
        _, fc, ec = col[b["kind"]]
        ax.add_patch(Rectangle(((b["x"] - x0) * MM, (b["y"] - y0) * MM), b["w"] * MM, b["h"] * MM, fc=fc, ec="white", lw=0.6))
    W, H = (x1 - x0) * MM, (y1 - y0) * MM
    ax.add_patch(Rectangle((0, 0), W, H, fill=False, ec=INK2, lw=1))
    arr = [b for b in bl if b["kind"] in ("ar", "cd") and b["pch"] == 0]             # one pseudo channel: 16 cells
    px0, py0 = (min(b["x"] for b in arr) - x0) * MM, (min(b["y"] for b in arr) - y0) * MM
    px1, py1 = (max(b["x"] + b["w"] for b in arr) - x0) * MM, (max(b["y"] + b["h"] for b in arr) - y0) * MM
    ax.add_patch(Rectangle((px0, py0), px1 - px0, py1 - py0, fill=False, ec=INK, lw=1.4, ls="--"))
    for (xx, yy, t) in ((W * 0.22, H * 0.985, "channel 0 (far side)\n32 cells, 64 banks"),
                        (W * 0.78, H * 0.985, "channel 1 (GPU side)\n32 cells, 64 banks")):
        ax.text(xx, yy, t, ha="center", va="top", fontsize=7, color=INK, bbox=dict(fc="white", ec="none", pad=1, alpha=0.9))
    ax.text((px0 + px1) / 2, (py0 + py1) / 2, "pseudo channel\n16 cells, 32 banks", ha="center", va="center", fontsize=7,
            color=INK, bbox=dict(fc="white", ec="none", pad=1.5, alpha=0.92))
    ax.annotate("", xy=(W + 1.7, H / 2), xytext=(W + 0.2, H / 2), arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.2))
    ax.text(W + 0.95, H / 2 + 0.3, "GPU\ndie", ha="center", va="bottom", fontsize=8, color=INK)
    handles = [Rectangle((0, 0), 1, 1, fc=c[1], ec=c[2]) for c in col.values()]
    ax.legend(handles, [c[0] for c in col.values()], loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=2, frameon=False, fontsize=7.5)
    ax.set_xlim(-0.3, W + 2.2)
    ax.set_ylim(-0.3, H + 0.3)
    ax.set_aspect("equal")
    ax.set_xlabel("mm")
    ax.set_title(f"DRAM die (stack 0, all 8 dies alike)\n128 banks of two HBM3 channels as 64 two-bank cells;\n"
                 f"bank arrays {int(a * 100)} % of the die", loc="left")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def beta_model():
    """(k, r_convec, operating point) of out/beta.json: the x = 0 point of the recalibration and the H100 GPU+PIM point
    at the threshold that beta gives."""
    if not os.path.exists(paths.out("beta.json")):
        raise SystemExit("no out/beta.json: run python -m thermal quick (or full) first")
    b = json.load(open(paths.out("beta.json")))
    return b["model"]["k"], b["model"]["r_convec"], dict(b["summary"]["operating_point"], beta=b["summary"]["every_bank"])


def unit_fields(grid):
    """Unit responses (K per W; GPU per W, the others per W per stack) of the x = 0 fit of the recalibration."""
    k, r, _ = beta_model()
    u = {}
    for s in ("gpu", "uni", "array", "edge"):
        F = HS.fields(f"{G.tag(k, r, grid, 'a50_par')}_{s}")
        u[s] = {key: v / P.amplitude(s) for key, v in F.items()}
    return u


def fields_at_op(grid):
    """PIM-off and PIM-on fields (degC) at the operating point where beta is evaluated, on the x = 0 model of the
    recalibration (memory heat in the DRAM dies; PIM at every bank with the MAC share at the column decoders)."""
    _, _, op = beta_model()
    u = unit_fields(grid)
    gpu_only = {key: G.AMB + op["P_G"] * u["gpu"][key] + op["memory_heat"] / G.STACKS * u["uni"][key] for key in u["gpu"]}
    pim = {key: gpu_only[key] + op["P_P"] / G.STACKS * ((1 - P.E_TABLE) * u["array"][key] + P.E_TABLE * u["edge"][key])
           for key in u["gpu"]}
    return gpu_only, pim, op


def design_points():
    """H100 GPU-only decode and GPU+PIM at the threshold (no power constraint) of out/beta.json, with the memory power
    P_M, the memory heat m0 + phi (P_M - m0) and, for GPU+PIM, the PIM power while it runs and its share of the step.
    Running power per stack = (8 e + MAC) pJ/B x PIM bandwidth at tCCD_L; share of the step = step-average PIM power /
    (stacks x running power x busy factor) (data/inputs/gpu_side_calibration.json)."""
    from .model import beta as B
    _, _, op = beta_model()
    o = B.gpu("H100")
    cal = {e["name"]: e["value"] for e in json.load(open(paths.inputs("gpu_side_calibration.json")))["inputs"]}
    p_run = (8 * op["e_pJ_bit"] + cal["MAC"]) * cal["bw_timing"]          # pJ/B x TB/s = W per stack
    go = dict(P_G=o["gpu_only"]["P_G"], P_M=o["gpu_only"]["P_M"], memory_heat=float(B.meff("H100", o["gpu_only"]["P_M"])),
              P_P=0.0)
    pim = dict(op, P_run_stack=p_run, duty=op["P_P"] / G.STACKS / (p_run * cal["busy_factor"]))
    return go, pim


def main_grid():
    k, r, _ = beta_model()
    for grid in (128, 64):
        if all(HS.done(f"{G.tag(k, r, grid, 'a50_par')}_{s}") for s in ("gpu", "uni", "array", "edge")):
            return grid
    raise SystemExit("no runs of the beta step: run python -m thermal quick (or full) first")


def _hot_marker(ax, F, ext, mask=None):
    """Mark and label the hottest grid cell of F (inside mask if given)."""
    n = F.shape[0]
    G_ = np.where(mask, F, -np.inf) if mask is not None else F
    i, j = np.unravel_index(np.argmax(G_), F.shape)
    x = ext[0] + (j + 0.5) * (ext[1] - ext[0]) / n
    y = ext[3] - (i + 0.5) * (ext[3] - ext[2]) / n
    ax.plot([x], [y], marker="x", ms=8, mew=2.6, color="white", zorder=5, clip_on=False)
    ax.plot([x], [y], marker="x", ms=7, mew=1.3, color=INK, zorder=6, clip_on=False)
    return x, y


def heatmap(path):
    """Hottest DRAM die: (a) GPU-only decode, (b) GPU+PIM at the H100 threshold, both as rise above ambient, (c) b - a."""
    grid = main_grid()
    k, r, _ = beta_model()
    u = unit_fields(grid)
    go, gp = design_points()

    def rise(p):
        return {key: p["P_G"] * u["gpu"][key] + p["memory_heat"] / G.STACKS * u["uni"][key]
                + p["P_P"] / G.STACKS * ((1 - P.E_TABLE) * u["array"][key] + P.E_TABLE * u["edge"][key]) for key in u["gpu"]}
    ra, rb = rise(go), rise(gp)
    mask = R_inside(grid)
    hot = max(range(8), key=lambda d: max(ra[f"L{4 + 2 * d}"][mask].max(), rb[f"L{4 + 2 * d}"][mask].max()))
    L = f"L{4 + 2 * hot}"
    A, Bf = ra[L], rb[L]
    D = Bf - A
    ta, tb = float(A[mask].max()), float(Bf[mask].max())
    ext = [0, G.W * MM, 0, G.H * MM]
    hi = np.ceil(max(A.max(), Bf.max()))
    levels = np.arange(5, hi, 5)
    fig = plt.figure(figsize=(12.9, 3.9))
    gs = fig.add_gridspec(1, 6, width_ratios=[1, 1, 0.04, 0.30, 1, 0.04], wspace=0.12)
    axs = [fig.add_subplot(gs[0, i]) for i in (0, 1, 4)]
    gpu_lab = "silicon above\nthe GPU die"
    fp, _ = G.footprints()
    titles = ("(a) GPU-only decode", "(b) GPU+PIM at the threshold")
    for ax, F, t in ((axs[0], A, titles[0]), (axs[1], Bf, titles[1])):
        im = ax.imshow(F, extent=ext, cmap=HEAT, vmin=0, vmax=hi, interpolation="nearest")
        cs = ax.contour(F, levels=levels, extent=ext, origin="upper", colors=INK, linewidths=0.5, alpha=0.55)
        for tx in ax.clabel(cs, fmt="%d K", fontsize=6.5, inline_spacing=3):
            px, py = tx.get_position()                 # keep labels off the narrow stack columns
            if any(x0 * MM - 1 <= px <= x1 * MM + 1 for x0, _y0, x1, _y1 in fp.values()):
                tx.set_visible(False)
        outlines(ax, label=True, gpu_label=gpu_lab)
        _hot_marker(ax, F, ext, mask)
        ax.set_title(f"{t}\nhottest DRAM cell +{F[mask].max():.1f} K (marked ×)", loc="left")
    cb = fig.colorbar(im, cax=fig.add_subplot(gs[0, 2]))
    cb.set_label("rise above ambient (K), (a) and (b)")
    lim = float(np.ceil(np.abs(D).max()))
    im2 = axs[2].imshow(D, extent=ext, cmap=DIVERGE, vmin=-lim, vmax=lim, interpolation="nearest")
    outlines(axs[2], label=True, color=INK, gpu_label=gpu_lab)
    axs[2].set_title(f"(c) difference, (b) minus (a)\nhottest DRAM cell {tb - ta:+.1f} K\nstack cells {D[mask].min():+.1f} "
                     f"to {D[mask].max():+.1f} K", loc="left")
    cb2 = fig.colorbar(im2, cax=fig.add_subplot(gs[0, 5]))
    cb2.set_label("(b) minus (a) (K)")
    pad = 0.6
    for ax in axs:
        ax.set_xlabel("position (mm)")
        ax.set_xlim(-pad, G.W * MM + pad)
        ax.set_ylim(-pad, G.H * MM + pad)
        ax.tick_params(labelsize=7)
    axs[0].set_ylabel("position (mm)")
    for ax in axs[1:]:
        ax.set_yticklabels([])
    fig.suptitle(f"H100 at the threshold: GPU-only and GPU+PIM are equally hot ({ta:+.1f} / {tb:+.1f} K, DRAM die {hot + 1} of 8, top view)",
                 x=0.125, ha="left", y=1.0, fontsize=11, color=INK)
    st = G.STACKS
    fit_c = json.load(open(paths.out("measured.json")))["h100_fit"]["c"] if os.path.exists(paths.out("measured.json")) else None
    cap = (f"PIM adds {gp['P_P'] / st:.1f} W per stack on average ({gp['P_run_stack']:.0f} W for {100 * gp['duty']:.0f} % of the step), and the KV reads leave the "
           f"off-chip path, so the heat in the stack drops from {go['memory_heat'] / st:.1f} to {gp['memory_heat'] / st:.1f} W per stack.\n"
           f"(a) GPU-only decode: GPU {go['P_G']:.0f} W, memory {go['P_M'] / st:.1f} W per stack "
           f"(heat in the stack {go['memory_heat'] / st:.1f} W), no PIM.\n"
           f"(b) GPU+PIM at the threshold ({gp['e_pJ_bit']:.2f} pJ/bit in-bank read energy, β = {gp['beta']:.2f}), no "
           f"power constraint: GPU {gp['P_G']:.0f} W, memory {gp['P_M'] / st:.1f} W per stack (heat in the stack "
           f"{gp['memory_heat'] / st:.1f} W; the KV reads stay in the banks),\n"
           f"      PIM {gp['P_P'] / st:.1f} W per stack as a step average: {gp['P_run_stack']:.0f} W per stack while it runs, "
           f"for {100 * gp['duty']:.0f} % of the step.\n"
           f"The threshold is the in-bank read energy at which the hottest DRAM cell of GPU+PIM reaches that of GPU-only, "
           f"so the two are equal by definition (here {tb - ta:+.1f} K: the threshold uses the measured slopes, the map the "
           f"HotSpot fit).\n"
           f"Absolute temperature = ambient + rise. The model uses a 50 °C ambient"
           + (f"; the measured H100 fit has intercept {fit_c:.1f} °C (PARAMETERS M03)" if fit_c is not None else "")
           + f". Contours every 5 K.\n"
           f"Model: x = 0 fit of the recalibration (memory heat in the DRAM dies), k = {k:.3f} W/m-K, r_convec {r:.5f} K/W, "
           f"grid {grid} x {grid}, steady state; PIM in the bank arrays ({100 * (1 - P.E_TABLE):.1f} %) and column decoders "
           f"({100 * P.E_TABLE:.1f} %).\n"
           f"Solid boxes: HBM stacks and, on this layer, the silicon column above the GPU die; dashed box: dark site "
           f"(unpowered).")
    fig.text(0.125, -0.02, cap, ha="left", va="top", fontsize=7.5, color=INK2, linespacing=1.5)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return dict(grid=grid, hottest_die=hot + 1, max_gpu_only=ta, max_gpu_pim=tb, diff=tb - ta,
                pim_duty=gp["duty"], pim_run_W_stack=gp["P_run_stack"])


def R_inside(grid):
    """Grid cells inside the five stack footprints (the HBM readout region)."""
    from .model import readout as RD
    return RD.readout(grid).inside


def stack_layers(path):
    grid = main_grid()
    _, g1, op = fields_at_op(grid)
    fp, _ = G.footprints()
    n = grid
    xc = (np.arange(n) + 0.5) * G.W / n
    yc = G.H - (np.arange(n) + 0.5) * G.H / n
    hs = max(fp, key=lambda s: max(g1[f"L{4 + 2 * d}"][np.ix_((yc > fp[s][1]) & (yc < fp[s][3]), (xc > fp[s][0]) & (xc < fp[s][2]))].max()
                                   for d in range(8)))
    x0, y0, x1, y1 = fp[hs]
    rows, cols_ = (yc > y0) & (yc < y1), (xc > x0) & (xc < x1)
    layers = [("base die", "L2")] + [(f"DRAM die {d + 1}", f"L{4 + 2 * d}") for d in range(8)]
    crops = [g1[L][np.ix_(rows, cols_)] - G.AMB for _, L in layers]
    lo, hi = min(c.min() for c in crops), max(c.max() for c in crops)
    fig, axs = plt.subplots(1, 9, figsize=(11.5, 2.9), sharey=True)
    ext = [x0 * MM, x1 * MM, y0 * MM, y1 * MM]
    for ax, (t, _), c in zip(axs, layers, crops):
        im = ax.imshow(c, extent=ext, cmap=HEAT, vmin=lo, vmax=hi, interpolation="nearest")
        ax.set_title(f"{t}\nmax +{c.max():.1f} K", fontsize=8, loc="left")
        ax.set_xticks([round(x0 * MM), round(x1 * MM)])
        ax.set_xlabel("x (mm)", fontsize=7)
        ax.tick_params(labelsize=7)
    axs[0].set_ylabel("y (mm)")
    side = "right" if (x0 + x1) / 2 < G.GX0 else "left"
    cb = fig.colorbar(im, ax=axs, fraction=0.012, pad=0.01, shrink=0.8)
    cb.set_label("rise above ambient (K)")
    fig.suptitle(f"Hottest stack (stack {hs}), GPU+PIM at the threshold, base die and DRAM dies 1-8 (bottom to top); the GPU die is to the {side}", x=0.125, ha="left",
                 y=0.98, fontsize=10, color=INK)
    k, r, _ = beta_model()
    fig.text(0.125, 0.0, f"Model: x = 0 fit of the recalibration (memory heat in the DRAM dies), k = {k:.3f} W/m-K, r_convec "
             f"{r:.5f} K/W, grid {grid} x {grid}, steady state; the same model and operating point as heatmap panel (b).\n"
             f"Operating point: H100 GPU+PIM at the threshold at \u03b2 = {op['beta']:.2f} ({op['e_pJ_bit']:.2f} pJ/bit), "
             f"PIM {op['P_P']:.1f} W over 5 stacks at every bank (step average). Absolute temperature = ambient (50 \u00b0C in the model) + rise.",
             ha="left", va="top", fontsize=7.5, color=INK2, linespacing=1.5)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return dict(stack=hs)


def calibration(path):
    res = json.load(open(paths.out("recal.json")))
    fam = [r for r in res["family"]["cell"] if r.get("x") is not None]
    x = np.array([r["x"] for r in fam])
    ratio = np.array([r["ratio"] for r in fam])
    beta = np.array([r["beta_eqv_every_bank"] for r in fam])
    t, ci = res["targets"]["ratio"], res["targets_ci95"]["ratio"]
    s = res["summary"]
    al = s["allowed"]["x"]
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(6.4, 6.0), sharex=True, gridspec_kw=dict(hspace=0.1))
    lo, hi = min(x.min(), 0) - 0.08, max(x.max(), 1) + 0.08
    for ax in (a1, a2):
        ax.axvspan(lo, 0, color="#f0efec", lw=0)
        ax.axvspan(1, hi, color="#f0efec", lw=0)
        ax.axvspan(al[0], al[1], color="#cde2fb", lw=0)
        ax.grid(color=GRIDC, lw=0.6)
        ax.set_axisbelow(True)
        ax.set_xlim(lo, hi)
    a1.axhspan(ci[0], ci[1], color="#fbd9c9", lw=0)
    a1.axhline(t, color=ORANGE, lw=1.2)
    a1.plot(x, ratio, color=BLUE, lw=2, marker="o", ms=3.5, zorder=3)
    a1.set_ylabel("row-miss heating ratio")
    a2.plot(x, beta, color=BLUE, lw=2, marker="o", ms=3.5, zorder=3)
    a2.set_ylabel("\u03b2, PIM at every bank")
    pts = [("x = 0: memory W all in the DRAM dies", 0.0, s["x0"]["ratio"], s["x0"]["beta"], "s"),
           ("reference only, not in the paper: our HBM3E split", s["prior"]["x"], s["prior"]["ratio"],
            s["prior"]["beta"], "D"),
           ("x that matches the measured ratio", s["best"]["x"], s["best"]["ratio"], s["best"]["beta"], "o"),
           ("x = 1: memory W all in the base die", 1.0, s["x1"]["ratio"], s["x1"]["beta"], "^")]
    hs = []
    for lab, xx, rr, bb, mk in pts:
        a1.plot([xx], [rr], marker=mk, ms=8, mfc="white", mec=INK, mew=1.3, ls="none", zorder=4)
        hs.append(a2.plot([xx], [bb], marker=mk, ms=8, mfc="white", mec=INK, mew=1.3, ls="none", zorder=4,
                          label=f"{lab}: \u03b2 = {bb:.2f}")[0])
    from matplotlib.patches import Patch
    a1.legend([Patch(fc="#fbd9c9"), Patch(fc="#cde2fb"), Patch(fc="#f0efec", ec=MUTED, lw=0.6)],
              [f"measured ratio {t:.2f}, 95 % CI {ci[0]:.2f}-{ci[1]:.2f}", "x whose HotSpot ratio is inside the CI",
               "x outside 0-1"], loc="upper right", fontsize=7.5, frameon=False)
    a2.legend(handles=hs, loc="upper right", fontsize=7.5, frameon=False)
    a2.set_xlabel("base-die share x of memory power")
    a1.set_title("Recalibration: every point matches both measured slopes", loc="left")
    fig.text(0.125, 0.015, "Each point is one bond-layer conductivity k (" + ", ".join(f"{r['k']:g}" for r in fam) +
             " W/m-K)\nwith its fitted x and heat-sink resistance. \u03b2: memory power that heats the hottest DRAM cell as "
             "much as the PIM power,\nper PIM W, at the H100 GPU+PIM threshold point that this \u03b2 gives.",
             fontsize=7, color=INK2, ha="left", va="top")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main(argv):
    out = argv[argv.index("--out") + 1] if "--out" in argv else os.path.join(os.path.dirname(paths.ROOT), "figures")
    os.makedirs(out, exist_ok=True)
    package(os.path.join(out, "thermal_package.png"))
    dram_die(os.path.join(out, "thermal_dram_die.png"))
    info = {}
    try:
        info["heatmap"] = heatmap(os.path.join(out, "thermal_heatmap.png"))
        info["stack"] = stack_layers(os.path.join(out, "thermal_stack_layers.png"))
    except SystemExit as e:
        print(f"[figures] heatmaps skipped: {e}")
    if os.path.exists(paths.out("recal.json")):
        calibration(os.path.join(out, "thermal_calibration.png"))
    else:
        print("[figures] calibration figure skipped: no out/recal.json")
    print(f"[figures] written to {out}: {sorted(f for f in os.listdir(out) if f.startswith('thermal_'))} {info}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
