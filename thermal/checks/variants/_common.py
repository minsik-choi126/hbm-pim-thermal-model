"""Shared paths and helpers of the HotSpot checks (HotSpot audit 2026-11-11).

Record: this file and the scripts next to it ran as <analysis>/pipeline/e2e_joint/hotspot_checks/ inside the analysis
repository (<analysis>) and do not run from this repository; their outputs are the JSON files in this folder.

Every check imports the repo model read only (hotspot_recal, thermal_bank_pim, run_sim2) and writes its HotSpot geometries
and unit runs under RUNS = <analysis>/pipeline/e2e_joint/runs_recal/hotspot_checks/; the small result
JSONs go next to the scripts. HotSpot binary: run_sim2.HOTSPOT (environment variable HOTSPOT overrides it).
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
E2E = os.path.dirname(HERE)                                  # <analysis>/pipeline/e2e_joint
ROOT = os.path.dirname(os.path.dirname(E2E))                 # repo root
RUNS = os.path.join(E2E, "runs_recal", "hotspot_checks")
AUDIT = os.path.join(HERE, "audit_copy")                     # result JSONs of the audit folder <audit>
if E2E not in sys.path:
    sys.path.insert(0, E2E)

import hotspot_recal as HR  # noqa: E402
import run_sim2 as R2  # noqa: E402

HOTSPOT = R2.HOTSPOT
TARGET_RATIO = HR.TARGET["ratio"]                            # measured row-miss ratio 0.70 (row-miss run result)
TARGET_RATIO_CI = HR.TARGET_CI["ratio"]

# fitted points of hotspot_recal.json solution.cell (x0: all memory power in the DRAM dies; best: the share that reaches
# the measured row-miss ratio; cur: the model before the recalibration)
POINTS = {
    "x0": dict(k=0.95369, r=0.01102, x=0.0),
    "best": dict(k=1.92697, r=0.01031, x=0.97511),
    "cur": dict(k=1.0, r=0.01361, x=0.0),
}


def runs_dir(*parts):
    d = os.path.join(RUNS, *parts)
    assert d.startswith(RUNS)
    os.makedirs(d, exist_ok=True)
    return d


def op_beta():
    """Operating point of beta: hotspot_recal.json op_beta (H100 GPU+PIM at its threshold, no power constraint)."""
    o = HR.J("hotspot_recal.json")["op_beta"]
    return {k: o[k] for k in ("P_G", "P_M", "dM", "P_P")}


def rnd(x, n=6):
    return HR.TP.rnd(x, n)


def dump(obj, name):
    p = os.path.join(HERE, name)
    json.dump(rnd(obj), open(p, "w"), indent=1)
    return p
