"""Measured HBM and GPU time constants of the H100 GEMM power step (data/measured/h100_exp1/h100_transient_dt2_r*.csv,
2 s sampling): first sample at or above 63 % of the heat-phase rise (end = mean of the last 30 s) and the share of the
rise still missing at 60 s. Compared with the model step response in trans_cmp_match.json (S08).

    python3 -m thermal.checks.validation.meas_t63      -> thermal/checks/validation/meas_t63.json
"""
import csv
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(HERE, "..", "..", "data", "measured", "h100_exp1")
out = {}
for i in (1, 2, 3):
    f = f"h100_transient_dt2_r{i}.csv"
    R = [r for r in csv.DictReader(open(os.path.join(D, f))) if r["phase"] == "heat"]
    t = np.array([float(r["t_s"]) for r in R])
    rec = {}
    for key in ("thbm", "tgpu"):
        T = np.array([float(r[key]) for r in R])
        s = (T - T[0]) / (T[t >= t[-1] - 30].mean() - T[0])
        rec[key] = dict(t63_first_sample_at_or_above=float(t[np.argmax(s >= 1 - np.exp(-1))]),
                        share_missing_at_60s=float(1 - s[np.argmin(np.abs(t - 60))]))
    out[f] = rec
json.dump(out, open(os.path.join(HERE, "meas_t63.json"), "w"), indent=1)
print(json.dumps(out, indent=1))
