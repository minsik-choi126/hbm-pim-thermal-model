"""File locations. Environment overrides: HOTSPOT (binary), THERMAL_RUNS (HotSpot run cache), THERMAL_OUT (results),
WORKERS (parallel HotSpot runs, default 8)."""
import os

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(ROOT, "data")
MEASURED = os.path.join(DATA, "measured")
INPUTS = os.path.join(DATA, "inputs")
GEOM_SRC = os.path.join(ROOT, "geometry", "h100_5stack")
CONFIG = os.path.join(ROOT, "geometry", "h100_5stack.config")
HOTSPOT_SRC = os.path.join(ROOT, "hotspot")
HOTSPOT = os.environ.get("HOTSPOT", os.path.join(HOTSPOT_SRC, "hotspot"))
RUNS = os.environ.get("THERMAL_RUNS", os.path.join(ROOT, "runs"))
OUT = os.environ.get("THERMAL_OUT", os.path.join(ROOT, "out"))
RESULTS = os.path.join(ROOT, "results")
WORKERS = int(os.environ.get("WORKERS", "8"))


def out(name):
    os.makedirs(OUT, exist_ok=True)
    return os.path.join(OUT, name)


def inputs(name):
    return os.path.join(INPUTS, name)
