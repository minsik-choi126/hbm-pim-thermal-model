"""Paths shared by the dram scripts (all relative to this directory; environment variables override)."""
import os

DRAM = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(DRAM, "data")
EXTERNAL = os.path.join(DRAM, "external")
# committed outputs; `run.sh full` writes to build/full instead and compares
RESULTS = os.environ.get("DRAM_RESULTS", os.path.join(DRAM, "results"))
REFERENCE = os.path.join(DRAM, "results")
ATTACC = os.environ.get("ATTACC_DIR", os.path.join(EXTERNAL, "attacc_simulator"))
RAM = os.path.join(ATTACC, "ramulator2")                 # AttAcc's Ramulator 2 tree (binary, trace generator)
RAMULATOR = os.path.join(RAM, "ramulator2")
HBM_POWER = os.environ.get("HBM_POWER_DIR", os.path.join(EXTERNAL, "HBM-Power"))


def out(name):
    """Path of an output file in RESULTS (created if missing)."""
    os.makedirs(RESULTS, exist_ok=True)
    return os.path.join(RESULTS, name)


def ref(name):
    """Path of a committed result."""
    return os.path.join(REFERENCE, name)


def need(path, setup):
    if not os.path.exists(path):
        raise SystemExit(f"missing {path}: run bash dram/setup/{setup}.sh first")
    return path
