"""Measured H100 data (thermal/data/measured, see SOURCES.md).

h100_fit()       HBM temperature T = c + s_host (board - memory W) + s_rail (memory W) over the 89 settled points of
                 the seven H100 files (GEMM and memory duty sweeps), block bootstrap over files (2000 draws, seed 0);
                 the same for the GPU-die temperature.
anchor_slopes()  OLS over the 60 settled GEMM rows of the three anchor files: HBM temperature and memory W per board W
                 (the calibration target of the 2 x 2 package model).
rowmiss()        row-miss microbenchmark: streaming reads (seq) and reads that open a new row at every access (rowmiss)
                 at matched read rates. The streaming rail line P_M = i + e B + g FMA (seq rows) gives the memory power a
                 streaming read of the same rate would draw; X = P_M - that is the extra row-activation power.
                 T = c + a P_G + b S + b_x X; ratio = b_x / b. Bootstrap resamples matched pairs, idle rows and FMA-only
                 rows within their groups (2000 draws, seed 20261110). Pure Python, as in the measurement kit.
"""
import csv
import math
import os
import random
import re

import numpy as np

from .. import paths

H100_DIR = os.path.join(paths.MEASURED, "h100_exp1")
ROWMISS_CSV = os.path.join(paths.MEASURED, "h100_rowmiss_rows.csv")
PAT = re.compile(r"^h100_(anchor_r1|anchor_r2|anchor_r3desc|combo2d[a-z0-9_]*|memsweep[a-z0-9_]*)\.csv$")
STACKS = 5


# ---------------------------------------------------------------- H100 model-card fit
def load_h100(d=H100_DIR):
    rows, idle = [], []
    for f in sorted(os.listdir(d)):
        if not PAT.match(f) or "samples" in f:
            continue
        for r in csv.DictReader(open(os.path.join(d, f))):
            if r.get("settled", "1") != "1" or not r.get("mem_power_w"):
                continue
            p, m = float(r["power"]), float(r["mem_power_w"])
            rec = dict(file=f, mode=r["mode"], cap=r["cap_W"], host=p - m, rail=m, board=p, thbm=float(r["thbm"]),
                       tgpu=float(r["tgpu"]))
            (idle if r["mode"] == "idle" else rows).append(rec)
    return rows, idle


def fit(rows, y_key="thbm"):
    X = np.array([[1.0, r["host"], r["rail"]] for r in rows])
    y = np.array([r[y_key] for r in rows])
    b = np.linalg.lstsq(X, y, rcond=None)[0]
    return b[-2], b[-1], b[0], float(np.std(y - X @ b))


def h100_fit(reps=2000):
    rows, _ = load_h100()
    files = sorted({r["file"] for r in rows})
    sh, sr, c, sd = fit(rows)
    gh, gr, _, gsd = fit(rows, y_key="tgpu")
    rng = np.random.default_rng(0)
    bs = []
    for _ in range(reps):
        pick = rng.choice(files, len(files), replace=True)
        rr = [r for f in pick for r in rows if r["file"] == f]
        if len({(round(r["host"]), round(r["rail"])) for r in rr}) < 4:
            continue
        bs.append(fit(rr)[:2] + fit(rr, y_key="tgpu")[:2])
    bs = np.array(bs)
    pct = lambda v: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]   # noqa: E731
    return dict(n=len(rows), n_files=len(files), files=files, c=float(c), s_host=float(sh), s_rail=float(sr), resid_sd=sd,
                s_host_ci=pct(bs[:, 0]), s_rail_ci=pct(bs[:, 1]), b_stack=float(STACKS * sr),
                gpu_s_host=float(gh), gpu_s_rail=float(gr), gpu_s_host_ci=pct(bs[:, 2]), gpu_s_rail_ci=pct(bs[:, 3]),
                gpu_resid_sd=gsd, boot_draws=int(len(bs)))


def h100_points():
    rows, _ = load_h100()
    return dict(host=np.array([r["host"] for r in rows]), rail=np.array([r["rail"] for r in rows]),
                thbm=np.array([r["thbm"] for r in rows]), tgpu=np.array([r["tgpu"] for r in rows]), n=len(rows))


def anchor_slopes():
    rows = []
    for f in ("h100_anchor_r1.csv", "h100_anchor_r2.csv", "h100_anchor_r3desc.csv"):
        rows += [r for r in csv.DictReader(open(os.path.join(H100_DIR, f))) if r["mode"] == "gemm" and r.get("settled", "1") == "1"]
    P = np.array([float(r["power"]) for r in rows])
    X = np.column_stack([np.ones(len(P)), P])
    t = np.linalg.lstsq(X, np.array([float(r["thbm"]) for r in rows]), rcond=None)[0][1]
    m = np.linalg.lstsq(X, np.array([float(r["mem_power_w"]) for r in rows]), rcond=None)[0][1]
    return dict(n=len(rows), hbm_per_board_W=float(t), mem_per_board_W=float(m))


# ---------------------------------------------------------------- row-miss run (pure Python, as the kit)
def _num(v):
    if v is None or v == "":
        return None
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def _truthy(v):
    return v in (True, "True", "true", 1, "1")


def _inv(A):
    n = len(A)
    M = [list(map(float, r)) + [1.0 if i == j else 0.0 for j in range(n)] for i, r in enumerate(A)]
    for c in range(n):
        p = max(range(c, n), key=lambda r: abs(M[r][c]))
        if abs(M[p][c]) < 1e-12:
            return None
        M[c], M[p] = M[p], M[c]
        piv = M[c][c]
        M[c] = [x / piv for x in M[c]]
        for r in range(n):
            if r != c and M[r][c] != 0.0:
                f = M[r][c]
                M[r] = [x - f * y for x, y in zip(M[r], M[c])]
    return [row[n:] for row in M]


def _ols(X, y):
    n, k = len(X), len(X[0]) if X else 0
    if n <= k:
        return None
    XtX = [[sum(r[i] * r[j] for r in X) for j in range(k)] for i in range(k)]
    Ai = _inv(XtX)
    if Ai is None:
        return None
    Xty = [sum(r[i] * v for r, v in zip(X, y)) for i in range(k)]
    return [sum(Ai[i][j] * Xty[j] for j in range(k)) for i in range(k)]


def rowmiss_rows(path=ROWMISS_CSV):
    """Valid and settled rows as fit records."""
    out = []
    for r in csv.DictReader(open(path)):
        if not (_truthy(r.get("valid")) and _truthy(r.get("settled"))):
            continue
        T, Bw, M = _num(r.get("thbm")), _num(r.get("board_w")), _num(r.get("mem_power_w"))
        if T is None or Bw is None or M is None:
            continue
        out.append(dict(kind=r.get("kind"), pat=r.get("pattern"), pair=r.get("pair") or "", fma=_num(r.get("fma")) or 0.0,
                        T=T, PM=M, PG=Bw - M, B=_num(r.get("mem_gbps")) or 0.0))
    return out


def _m0(d):
    idle = [x["PM"] for x in d if x["kind"] == "idle"]
    return sum(idle) / len(idle) if idle else min(x["PM"] for x in d)


def _rail_line(rows):
    y = [x["PM"] for x in rows]
    b = _ols([[1.0, x["B"], x["fma"]] for x in rows], y)
    if b is not None:
        return dict(i=b[0], e=b[1], g=b[2])
    b = _ols([[1.0, x["B"]] for x in rows], y)
    return dict(i=b[0], e=b[1], g=0.0) if b else None


def fit_excess(d):
    m0 = _m0(d)
    L = _rail_line([x for x in d if x["pat"] == "seq"])
    X, y, ex = [], [], []
    for x in d:
        if x["pat"] == "seq":
            S, E = x["PM"] - m0, 0.0
        elif x["pat"] == "rowmiss":
            base = L["i"] + L["e"] * x["B"] + L["g"] * x["fma"]
            S, E = base - m0, x["PM"] - base
            ex.append(E)
        else:
            S = E = 0.0
        X.append([1.0, x["PG"], S, E])
        y.append(x["T"])
    b = _ols(X, y)
    if b is None:
        return None
    a, bb, bx = b[1:4]
    return dict(a=a, b=bb, b_x=bx, ratio=(bx / bb) if bb else None, n=len(y), m0=m0,
                excess_mean_w=sum(ex) / len(ex) if ex else None, rail_seq=L)


def _units(d):
    by_pair, groups = {}, {"pair": [], "idle": [], "fma": [], "single": []}
    for x in d:
        if x["pair"] and x["pat"] in ("seq", "rowmiss"):
            by_pair.setdefault(x["pair"], []).append(x)
        elif x["kind"] == "idle":
            groups["idle"].append([x])
        elif x["kind"] == "fma":
            groups["fma"].append([x])
        else:
            groups["single"].append([x])
    for pid in sorted(by_pair):
        u = by_pair[pid]
        (groups["pair"] if len(u) == 2 else groups["single"]).append(u)
    return [g for g in groups.values() if g]


def _ci(v, reps):
    if len(v) < max(20, reps // 2):
        return dict(lo=None, hi=None, n=len(v))
    v = sorted(v)

    def q(p):
        i = p * (len(v) - 1)
        lo = int(math.floor(i))
        return v[lo] + (v[min(lo + 1, len(v) - 1)] - v[lo]) * (i - lo)
    return dict(lo=q(0.025), hi=q(0.975), n=len(v))


def rowmiss(reps=2000, seed=20261110):
    d = rowmiss_rows()
    main = fit_excess(d)
    rng = random.Random(seed)
    strata = _units(d)
    vals = []
    for b in range(reps):
        sample = []
        for units in strata:
            for j in range(len(units)):
                u = units[rng.randrange(len(units))]
                sample.extend(dict(x, pair=(x["pair"] + f"#{b}.{j}") if x["pair"] else "") for x in u)
        try:
            e = fit_excess(sample)
        except (ZeroDivisionError, ValueError, TypeError):
            continue
        if e is not None and e["ratio"] is not None and math.isfinite(e["ratio"]):
            vals.append(e["ratio"])
    return dict(n=len(d), main=main, boot_ratio=_ci(vals, reps), reps=reps, seed=seed)


def rowmiss_points():
    """Row-miss rows for the HotSpot replay: non-memory W (PG), memory W (PM), its streaming-equivalent part above idle
    (S) and the extra row-activation power (E), HBM temperature (T)."""
    d = rowmiss_rows()
    m0 = _m0(d)
    seq = [x for x in d if x["pat"] == "seq"]
    Xs = np.column_stack([np.ones(len(seq)), [x["B"] for x in seq], [x["fma"] for x in seq]])
    i_, e_, g_ = np.linalg.lstsq(Xs, np.array([x["PM"] for x in seq]), rcond=None)[0]
    S, E = [], []
    for x in d:
        if x["pat"] == "seq":
            S.append(x["PM"] - m0)
            E.append(0.0)
        elif x["pat"] == "rowmiss":
            base = i_ + e_ * x["B"] + g_ * x["fma"]
            S.append(base - m0)
            E.append(x["PM"] - base)
        else:
            S.append(0.0)
            E.append(0.0)
    out = dict(PG=np.array([x["PG"] for x in d]), PM=np.array([x["PM"] for x in d]), S=np.array(S), E=np.array(E),
               T=np.array([x["T"] for x in d]), m0=m0, n=len(d))
    out["E_mem_stream"] = out["PM"] - out["E"]          # streaming-equivalent memory power (idle included)
    return out
