"""Record: ran inside the analysis repository (<analysis>), not from this repository. GPU-die temperature slope of the calibrated HotSpot family against the measured one, a value the fit does not use
(HotSpot audit 11/11, the audit report 2.1 'GPU-die slope 15 % below measurement'; part (a) of script validation/e1_steady.py of
the audit). Analysis of the existing unit runs of hotspot_recal.py (runs_recal/runs, read only), no new HotSpot runs.

Along hotspot_recal.json family.cell (each k with the r_convec and base-die share x that match both measured HBM slopes),
HotSpot is evaluated at the 89 settled points of the H100 model-card fit (hotspot_recal.pts) and regressed like
paths_multi.fit; the GPU-die slope per W of non-memory power and per W of memory power, read three ways (hottest GPU-die
cell, die mean, cell nearest the die centre), at x = 0, at the Olgun et al. prior share, at the row-miss match and at x = 1
(linear in x along the family). Measured: paths_multi.json H100 gpu_s_host (per W of non-memory power) and gpu_s_rail.

Inputs: hotspot_recal.json, paths_multi.json, runs_recal/runs. Output: <analysis>/pipeline/e2e_joint/hotspot_checks/gpu_die.json (copied here).

    python3 <analysis>/pipeline/e2e_joint/hotspot_checks/gpu_die.py      # about 1 min
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _common as C  # noqa: E402

HR = C.HR
LABS = ("hbm_cell_max", "gpu_max", "gpu_mean", "gpu_centre")


def main():
    rec_ = HR.J("hotspot_recal.json")
    fam = [r for r in rec_["family"]["cell"] if r.get("x") is not None]
    H, _ = HR.pts()
    ro = HR.readout(64)
    n = ro.grid
    xc = (np.arange(n) + 0.5) * HR.TP.S.W / n
    yc = HR.TP.S.H - (np.arange(n) + 0.5) * HR.TP.S.H / n
    X, Y = np.meshgrid(xc, yc)
    d2 = (X - 0.5 * (HR.TP.S.GX0 + HR.TP.S.GX1)) ** 2 + (Y - 0.5 * (HR.TP.S.GY0 + HR.TP.S.GY1)) ** 2
    d2[~ro.gpu] = np.inf
    ci = np.unravel_index(np.argmin(d2), d2.shape)

    def unit(k, r):
        out = {}
        for s in ("gpu", "uni", "base"):
            F = dict(np.load(os.path.join(HR.OUT, "runs", f"{HR.gtag(k, r, 64)}_{s}", "fields.npz")))
            amp = HR.GPU_W if s == "gpu" else HR.UNIT_W
            out[s] = {kk: v / amp for kk, v in F.items()}
        return out

    def slopes(k, x, r):
        U = unit(k, r)
        w = {"gpu": H["host"], "uni": (1 - x) * H["rail"] / HR.STACKS, "base": x * H["rail"] / HR.STACKS}
        acc = dict.fromkeys(LABS)
        for s, ww in w.items():
            F = U[s]
            cells = np.concatenate([F[f"L{4 + 2 * d}"][ro.inside] for d in range(8)])
            parts = dict(hbm_cell_max=ww[:, None] * cells[None, :], gpu_max=ww[:, None] * F["L2"][ro.gpu][None, :],
                         gpu_centre=ww * F["L2"][ci])
            for lab, t in parts.items():
                acc[lab] = t if acc[lab] is None else acc[lab] + t
        acc["gpu_mean"] = acc["gpu_max"].mean(axis=1)
        acc["gpu_max"] = acc["gpu_max"].max(axis=1)
        acc["hbm_cell_max"] = acc["hbm_cell_max"].max(axis=1)
        m = {}
        for lab in LABS:
            b = HR.ols([H["host"], H["rail"]], acc[lab])
            m[f"{lab}_perG"], m[f"{lab}_perM"] = float(b[1]), float(b[2])
        return m

    def row(k, x, r):
        if r in HR.RS:
            return slopes(k, x, r)
        rows = [slopes(k, x, rr) for rr in HR.RS]
        return {key: float(np.polyval(np.polyfit(HR.RS, [q[key] for q in rows], 2), r)) for key in rows[0]}

    fam_rows = [dict(k=f["k"], x=f["x"], r_convec=f["r_convec"], **row(f["k"], f["x"], f["r_convec"])) for f in fam]
    meas = rec_["checks_measured"]

    def at_x(key, xv):
        xs = [q["x"] for q in fam_rows]
        for i in range(len(xs) - 1):
            if (xs[i] - xv) * (xs[i + 1] - xv) <= 0:
                t = (xv - xs[i]) / (xs[i + 1] - xs[i])
                return fam_rows[i][key] + t * (fam_rows[i + 1][key] - fam_rows[i][key])
        return None
    pts = {}
    for lab, xv in (("x0", 0.0), ("prior", rec_["prior_fit"]["x"]), ("rowmiss_match", rec_["best_fit"]["x"]), ("x1", 1.0)):
        p = dict(x=xv, **{key: at_x(key, xv) for key in ("k", "r_convec") + tuple(f"{a}_{b}" for a in LABS for b in ("perG", "perM"))})
        for lab2 in ("gpu_max", "gpu_mean", "gpu_centre"):
            p[f"{lab2}_perG_over_meas"] = p[f"{lab2}_perG"] / meas["gpu_die_per_nonmem_W"] - 1
        pts[lab] = p
    gm = [p["gpu_max_perG"] for p in pts.values()]
    res = dict(measured=dict(gpu_die_per_nonmem_W=meas["gpu_die_per_nonmem_W"], ci95=meas["gpu_die_per_nonmem_W_ci"],
                             gpu_die_per_mem_W=meas["gpu_die_per_mem_W"], gpu_die_per_mem_W_ci=meas["gpu_die_per_mem_W_ci"],
                             src="paths_multi.json H100 gpu_s_host, gpu_s_rail (hotspot_recal.json checks_measured)"),
               points=pts, family=fam_rows,
               gpu_max_perG_range=[min(gm), max(gm)],
               gpu_max_perG_over_meas_range=[min(gm) / meas["gpu_die_per_nonmem_W"] - 1,
                                             max(gm) / meas["gpu_die_per_nonmem_W"] - 1],
               hotspot_recal_best_fit_residual_gpu_die=rec_["best_fit"]["residual"]["gpu_die"])
    C.dump(res, "gpu_die.json")
    print({k: [round(v, 4) for v in res[k]] for k in ("gpu_max_perG_range", "gpu_max_perG_over_meas_range")})


if __name__ == "__main__":
    main()
