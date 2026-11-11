"""Export the GPU+PIM operating points that the beta fixed point needs -> thermal/data/inputs/beta_operating_points.json.

The heat factor beta is evaluated at the operating point of GPU+PIM at the H100 threshold that the threshold model gives
with that beta (thermal/steps/beta.py). The operating point as a function of the in-bank read energy e comes from the
GPU-side decode model of the analysis repository, which this repository does not contain. This script reads it there
(read only) and writes, per GPU (H100, H200, B200), on e = 0-6 pJ/bit in steps of 0.01:
    P_G, P_M, P_P   non-memory, memory (idle included) and PIM power (W, all stacks) of GPU+PIM decoding, no power
                    constraint (abs_threshold.point(g, e, "timing")["P_G" | "P_M" | "P_P"])
and the scalars
    gpu_only        P_G, P_M of GPU-only decoding (abs_threshold.GO[g]["split"])
    m0              idle memory power (abs_threshold.IDLE[g][1])
    phi             share of memory power above idle that heats the stacks (abs_threshold.CARD[g]["rail_heat"])
    a, b            measured HBM rise per non-memory W and per memory W (abs_threshold.coef_model(g))
Threshold model (abs_threshold.dT_lin): dT = a (P_G - P_G0) + b (M - M0) + b beta P_P, memory heat M = m0 + phi (P_M - m0).
At dc 0 the operating points do not depend on beta (checked below).

    python3 tools/export_beta_ops.py <analysis repository root or a snapshot of it>
"""
import json
import os
import re
import subprocess
import sys

E_STEP, E_MAX = 0.01, 6.0
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "thermal", "data", "inputs",
                   "beta_operating_points.json")


def main(root):
    here = os.path.join(root, "pipeline", "e2e_joint")
    sys.path.insert(0, here)
    import abs_threshold as A  # noqa: E402
    es = [round(E_STEP * i, 2) for i in range(int(round(E_MAX / E_STEP)) + 1)]
    assert set(es) <= set(A.E_FINE)
    commit = os.environ.get("ANALYSIS_COMMIT") or subprocess.run(
        ["git", "-C", root, "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip() or None
    gpus = {}
    for g in A.GPUS:
        pts = [A.point(g, e, "timing") for e in es]
        for e in (1.0, 2.0, 3.0):                       # dc 0: the operating point does not depend on beta
            p0, p1 = A.point(g, e, "timing"), A.point(g, e, "timing", f=0.6)
            assert all(abs(p0[k] - p1[k]) < 1e-9 for k in ("P_G", "P_M", "P_P")), (g, e)
        go, km = A.GO[g]["split"], A.coef_model(g)
        gpus[g] = dict(gpu_only=dict(P_G=go["P_G"], P_M=go["P_M"]), m0=A.IDLE[g][1], phi=km["phi"], a=km["a"], b=km["b"],
                       P_G=[p["P_G"] for p in pts], P_M=[p["P_M"] for p in pts], P_P=[p["P_P"] for p in pts])
    out = dict(
        note="GPU+PIM decoding operating points (no power constraint, dc 0) over the in-bank read energy e, per GPU, for the "
             "beta fixed point of thermal/steps/beta.py. Powers in W over all stacks; P_M includes idle memory power; memory "
             "heat of the threshold model M = m0 + phi (P_M - m0). a, b: measured HBM rise per non-memory W and per memory W "
             "(K/W). Not recomputed here.",
        source=dict(repository="analysis repository (private)", commit=commit, script="pipeline/e2e_joint/abs_threshold.py",
                    functions=dict(P_G_P_M_P_P='point(g, e, "timing")', gpu_only='GO[g]["split"]', m0="IDLE[g][1]",
                                   phi='CARD[g]["rail_heat"]', a_b="coef_model(g)"),
                    exporter="tools/export_beta_ops.py"),
        e_pJ_bit=dict(start=0.0, step=E_STEP, stop=E_MAX, n=len(es)), gpus=gpus)
    txt = json.dumps(out, indent=1)
    txt = re.sub(r"\[\s*([-0-9.e,\s]+?)\s*\]", lambda m: "[" + ", ".join(m.group(1).split()).replace(",,", ",") + "]", txt)
    with open(OUT, "w") as f:
        f.write(txt + "\n")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main(os.path.abspath(sys.argv[1]))
