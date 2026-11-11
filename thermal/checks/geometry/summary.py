"""Per variant: beta at x = 0 and x = 1, the x where the HotSpot row-miss ratio meets the measured 0.701 (linear in x
between the two ends, clamped to [0, 1]), beta there, and the H100 threshold (hotspot_recal.json threshold_sweep, no
power constraint) at those betas. Reads refit_result.json (analyze2.py)."""
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
R = json.load(open(os.path.join(HERE, "refit_result.json")))
REC = json.load(open("<analysis>/pipeline/e2e_joint/hotspot_recal.json"))
SW = REC["threshold_sweep"]
TGT, CI_HI = 0.70109, 0.77851


def thr(beta, g="H100"):
    b = np.array(SW["beta"])
    e = np.array([np.nan if v is None else v for v in SW[g]["timing"]])
    return float(np.interp(beta, b, e))


out = {}
for var, rec in R.items():
    if var == "base_rows" or "x0" not in rec:
        continue
    s = dict(beta_x0=rec["x0"]["beta_array_only"], gpu_die_x0=rec["x0"]["gpu_die_per_nonmem_W"], ratio_x0=rec["x0"]["ratio"],
             k_x0=float(rec["x0"]["k_star"]), r_x0=rec["x0"].get("r_equiv"), thr_x0=thr(rec["x0"]["beta_array_only"]))
    if "x1" in rec:
        b0, b1, q0, q1 = rec["x0"]["beta_array_only"], rec["x1"]["beta_array_only"], rec["x0"]["ratio"], rec["x1"]["ratio"]
        xs = (q0 - TGT) / (q0 - q1)
        xc = min(max(xs, 0.0), 1.0)
        xhi = min(max((q0 - CI_HI) / (q0 - q1), 0.0), 1.0)
        s.update(beta_x1=b1, ratio_x1=q1, k_x1=float(rec["x1"]["k_star"]), gpu_die_x1=rec["x1"]["gpu_die_per_nonmem_W"],
                 x_match_unclamped=xs, x_match=xc, beta_match=b0 + xc * (b1 - b0), x_ci_hi=xhi, beta_ci_hi=b0 + xhi * (b1 - b0))
        s["thr_match"] = thr(s["beta_match"])
    out[var] = s
base = out["base"]
for var, s in out.items():
    s["dbeta_x0"] = s["beta_x0"] - base["beta_x0"]
    if "beta_match" in s:
        s["dbeta_match"] = s["beta_match"] - base["beta_match"]
        s["dthr_match"] = s["thr_match"] - base["thr_match"]
    s["dthr_x0"] = s["thr_x0"] - base["thr_x0"]
json.dump(out, open(os.path.join(HERE, "summary_result.json"), "w"), indent=1)
for var, s in out.items():
    print(var.ljust(10), " ".join(f"{k}={v:.4f}" for k, v in s.items() if isinstance(v, float)))
