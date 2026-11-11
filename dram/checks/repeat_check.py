"""Convergence of the refresh slowdown in the trace length (REPEAT of refresh_sweep.py).

Runs the 2x / 1x refresh ratio for REPEAT 4, 8 (the sweep's value) and 16 at H100 (5.2 Gb/s, tRFC 350 ns, nCCDAB 6)
and B200 (8.0 Gb/s, tRFC 410 ns, nCCDAB 9), for all-bank refresh and both per-bank modes.

python -m dram.checks.repeat_check          # output: dram/checks/out/repeat_check.txt
"""
import os
import tempfile
from concurrent.futures import ThreadPoolExecutor

from dram.attacc import refresh_sweep as RS

CASES = [("HBM3_5.2Gbps", 350, 6), ("HBM3_8.0Gbps", 410, 9)]
MODES = ("1x", "2x", "pbj1x", "pbj2x", "pbh1x", "pbh2x")


def main():
    for rep in (4, 8, 16):
        RS.REPEAT = rep
        with tempfile.TemporaryDirectory() as tmp:
            tr = os.path.join(tmp, "t.trace")
            RS.make_trace(tr)
            jobs = [(p, t, n, m, tr, tmp) for p, t, n in CASES for m in MODES]
            with ThreadPoolExecutor(8) as ex:
                rows = list(ex.map(RS.run, jobs))
        d = {(r["preset"], r["refresh"]): r["cycles"] for r in rows}
        for p, t, n in CASES:
            print(f"REPEAT {rep:2d} {p} nCCDAB {n}: 2x/1x all-bank {d[(p, '2x')] / d[(p, '1x')]:.4f}, "
                  f"pbj {d[(p, 'pbj2x')] / d[(p, 'pbj1x')]:.4f}, pbh {d[(p, 'pbh2x')] / d[(p, 'pbh1x')]:.4f}", flush=True)


if __name__ == "__main__":
    main()
