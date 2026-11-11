"""Shared setup: import the repo model read-only, redirect every write to this scratch folder."""
import os
import sys

os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
sys.dont_write_bytecode = True
SCR = os.path.dirname(os.path.abspath(__file__))
REPO = "<analysis>/pipeline/e2e_joint"
sys.path.insert(0, REPO)

import thermal_bank_pim as B  # noqa: E402
import hotspot_recal as HR  # noqa: E402

B.OUT = os.path.join(SCR, "runs_bank")       # bank geometries copied here; new grids built here
HR.OUT = os.path.join(SCR, "runs_recal")     # recal geometries and unit runs built here
assert HR.OUT.startswith(SCR) and B.OUT.startswith(SCR)
REPO_RUNS = os.path.join(REPO, "runs_recal", "runs")

# fitted points of hotspot_recal.json (solution.cell)
POINTS = {
    "x0": dict(k=0.95369, r=0.01102, x=0.0),
    "best": dict(k=1.92697, r=0.01031, x=0.97511),
    "cur": dict(k=1.0, r=0.01361, x=0.0),
}
