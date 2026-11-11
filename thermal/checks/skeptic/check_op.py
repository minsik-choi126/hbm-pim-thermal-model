"""Skeptic check 2: beta (every bank, cell readout, local normalization) at the operating point where each threshold
is read, versus the decode operating point used by hotspot_recal (e = 1.84 pJ/bit). Read-only use of runs_recal."""
import json
import os
import sys

REPO = "<analysis>/pipeline/e2e_joint"
sys.path.insert(0, REPO)
import hotspot_recal as HR  # noqa: E402
import abs_threshold as A  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
fam = [r for r in HR.J("hotspot_recal.json")["family"]["cell"] if r.get("x") is not None]
XT = HR.J("hotspot_recal.json")["solution"]["cell"]["target_ratio"]["x"]
out = {}
for g in ("H100", "H200", "B200"):
    for e in (1.8355, 2.5, 3.10):
        p = A.point(g, e, "timing")
        op = dict(P_G=p["P_G"], P_M=p["P_M"], dM=20.0, P_P=p["P_P"])
        rows = []
        for q in fam:
            r = HR.at(q["k"], q["x"], q["r_convec"], "cell", op_beta=op)
            rows.append(dict(k=q["k"], x=q["x"], **{kk: r[kk] for kk in ("beta_every_bank", "beta_every_bank_reg", "beta_every_bank_amp")}))
        rec = dict(op={k: round(v, 2) for k, v in op.items()})
        for lab, xv in (("x0", 0.0), ("match", XT), ("x1", 1.0)):
            rec[lab] = {kk: HR.interp_family(rows, kk, xv, "x") for kk in ("beta_every_bank", "beta_every_bank_reg", "beta_every_bank_amp")}
        out[f"{g}_e{e}"] = rec
        print(g, e, json.dumps(rec), flush=True)
json.dump(out, open(os.path.join(HERE, "op_result.json"), "w"), indent=1)
