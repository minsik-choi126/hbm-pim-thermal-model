"""Quick form of repeat_check.py: H100 (5.2 Gb/s, tRFC 350 ns, nCCDAB 6) all-bank 2x / 1x at REPEAT 4 and 16.

python -m dram.checks.rep4                  # output: dram/checks/out/rep4.txt
"""
import os
import tempfile
from concurrent.futures import ThreadPoolExecutor

from dram.attacc import refresh_sweep as RS


def main():
    for rep in (4, 16):
        RS.REPEAT = rep
        with tempfile.TemporaryDirectory() as tmp:
            tr = os.path.join(tmp, "t.trace")
            RS.make_trace(tr)
            jobs = [("HBM3_5.2Gbps", 350, 6, m, tr, tmp) for m in ("1x", "2x")]
            with ThreadPoolExecutor(2) as ex:
                r = list(ex.map(RS.run, jobs))
        print(f"REPEAT {rep:2d}: all-bank 2x/1x {r[1]['cycles'] / r[0]['cycles']:.4f}")


if __name__ == "__main__":
    main()
