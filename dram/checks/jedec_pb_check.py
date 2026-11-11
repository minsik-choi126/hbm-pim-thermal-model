"""Per-bank refresh: which JESD238 timing moves the 2x / 1x ratio, at each GPU's AttAcc MAC interval and main tRFC.

  pbh   HBM2-derived tRFCsb (160 / 187 ns), tRREFD per SID (comparison mode of refresh_sweep.py)
  sid   JESD238B.01 Table 93 tRFCpb (200 / 240 ns), tRREFD per SID
  pbj   JESD238B.01 tRFCpb and tRREFD per pseudo channel (Table 3): the main per-bank mode of refresh_sweep.py
Each against all-bank refresh at 2x (also relative to all-bank 1x).

python -m dram.checks.jedec_pb_check        # output: dram/checks/out/jedec_pb_check.txt
"""
import os
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor

from dram.attacc import refresh_sweep as RS

CASES = [("HBM3_5.2Gbps", 350, 6), ("HBM3_6.4Gbps", 410, 7), ("HBM3_8.0Gbps", 410, 9)]


def run(a):
    p, t, n, m, tr, tmp = a
    if m.startswith("sid"):                    # pbj tRFCpb with the per-SID tRREFD count
        yml = RS.config(p, t, n, m.replace("sid", "pbj"), tr).replace("    refsb_rrefd_level: pseudochannel\n", "")
        path = os.path.join(tmp, f"s_{p}_{m}.yaml")
        open(path, "w").write(yml)
        out = subprocess.run([os.path.join(RS.RAM, "ramulator2"), "-f", path], capture_output=True, text=True, check=True).stdout
        return dict(preset=p, refresh=m, cycles=int(next(l for l in out.splitlines() if "memory_system_cycles" in l).split()[-1]))
    return RS.run(a)


def main():
    modes = ("1x", "2x", "pbh1x", "pbh2x", "sid1x", "sid2x", "pbj1x", "pbj2x")
    with tempfile.TemporaryDirectory() as tmp:
        tr = os.path.join(tmp, "t.trace")
        RS.make_trace(tr)
        jobs = [(p, t, n, m, tr, tmp) for p, t, n in CASES for m in modes]
        with ThreadPoolExecutor(8) as ex:
            rows = list(ex.map(run, jobs))
    d = {(r["preset"], r["refresh"]): r["cycles"] for r in rows}
    for p, t, n in CASES:
        a1 = d[(p, "1x")]
        print(f"{p} nCCDAB {n} tRFC {t} ns: 2x/1x all-bank {d[(p, '2x')] / a1:.4f} | pbh {d[(p, 'pbh2x')] / d[(p, 'pbh1x')]:.4f}"
              f" | sid {d[(p, 'sid2x')] / d[(p, 'sid1x')]:.4f} | pbj {d[(p, 'pbj2x')] / d[(p, 'pbj1x')]:.4f}"
              f" || per-bank 2x over all-bank 1x: pbh {d[(p, 'pbh2x')] / a1:.4f}, sid {d[(p, 'sid2x')] / a1:.4f},"
              f" pbj {d[(p, 'pbj2x')] / a1:.4f}")


if __name__ == "__main__":
    main()
