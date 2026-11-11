"""nCCDL 5 (patch 02, JESD238B.01 Table 93 tCCDL = max(4 nCK, 2.5 ns) at 8.0 Gb/s) against 4 (AttAcc's fixed value):
PIM attention run time over the MAC sweep points (3 contexts x 9 MAC intervals) at 8.0 Gb/s.

python -m dram.checks.nccdl_check           # output: dram/checks/out/nccdl_check.txt
"""
import os
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor

sys.argv = sys.argv[:1] + ["HBM3_8.0Gbps"]
from dram.attacc import mac_sweep as M          # noqa: E402  (reads the preset from argv)
from dram.attacc import refresh_sweep as RS     # noqa: E402

P = "HBM3_8.0Gbps"


def run(a):
    L, n, nccdl, tmp = a
    y = M.YAML.format(trace=os.path.join(tmp, f"t_{L}.trace"), nccdab=n, preset=P, nrfc=RS.cycles(RS.MAIN_TRFC[P], P))
    if nccdl:
        y = y.replace("      nCCDAB:", f"      nCCDL: {nccdl}\n      nCCDAB:")
    f = os.path.join(tmp, f"c_{L}_{n}_{nccdl}.yaml")
    open(f, "w").write(y)
    out = subprocess.run([os.path.join(M.RAM, "ramulator2"), "-f", f], capture_output=True, text=True, check=True).stdout
    return L, n, nccdl, int([l for l in out.splitlines() if "memory_system_cycles" in l][0].split()[-1])


def main():
    with tempfile.TemporaryDirectory() as tmp:
        for L in M.SEQLENS:
            subprocess.run([sys.executable, os.path.join(M.RAM, "trace_gen", "gen_trace_attacc_bank.py"), "--dhead", "128",
                            "--nhead", "64", "--seqlen", str(L), "--dbyte", "2", "--output", os.path.join(tmp, f"t_{L}.trace")],
                           check=True, capture_output=True)
        jobs = [(L, n, c, tmp) for L in M.SEQLENS for n in M.NCCDAB for c in (0, 4)]
        with ThreadPoolExecutor(8) as ex:
            res = list(ex.map(run, jobs))
    d = {(L, n, c): cy for L, n, c, cy in res}
    diffs = []
    print("context nCCDAB cycles(nCCDL 5) cycles(nCCDL 4) change %")
    for L in M.SEQLENS:
        for n in M.NCCDAB:
            x = 100 * (d[(L, n, 0)] / d[(L, n, 4)] - 1)
            diffs.append(x)
            print(L, n, d[(L, n, 0)], d[(L, n, 4)], round(x, 3))
    print(f"nCCDL 5 vs 4, run time change over {len(diffs)} points: {min(diffs):.3f} to {max(diffs):.3f} %")


if __name__ == "__main__":
    main()
