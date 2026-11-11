"""PIM attention speed vs PIM power limit, from the AttAcc (ASPLOS'24) bank-level Ramulator2 model.

AttAcc models the HBM power constraint as the minimum interval between all-bank MAC commands (nCCDAB): its presets
use 6 cycles at 5.2 Gbps and 7 at 6.4 Gbps (computed from activation and read energy, AttAcc README); no power
constraint (the _NPC presets) is the DRAM timing limit, tCCDL = max(4 nCK, 2.5 ns) (JESD238B.01 Table 93;
refresh_sweep.N_TIMING): 4 cycles at 5.2 and 6.4 Gbps, 5 at 8.0 Gbps. The sweep starts there.
Organisation HBM3_8Gb_2R (AttAcc's) with the GPU's all-bank tRFC
(refresh_sweep.MAIN_TRFC). PIM MAC power scales with the MAC command rate
(1 / nCCDAB); attention time comes from the cycle-level run, so ACT/PRE, refresh and softmax overheads
that do not scale with the MAC rate are kept. mac_rate_rel and attn_speed_rel are relative to the timing limit.

python -m dram.attacc.mac_sweep HBM3_5.2Gbps   # writes dram/results/mac_sweep_HBM3_5.2Gbps.csv
python -m dram.attacc.mac_sweep HBM3_6.4Gbps   # H200
python -m dram.attacc.mac_sweep HBM3_8.0Gbps   # B200
"""
import csv
import os
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor

from dram import common as C
from dram.attacc import refresh_sweep as RS                             # tRFC of each data rate

RAM = C.RAM
# argv[1]: HBM3_5.2Gbps (AttAcc default, H100), HBM3_6.4Gbps (H200; AttAcc preset nCCDAB 7) or HBM3_8.0Gbps (B200 HBM3E;
# preset added by dram/patches/attacc/02, nCCDAB 9)
PRESET = (sys.argv[1:] or ["HBM3_5.2Gbps"])[0]
N_T = RS.N_TIMING[PRESET]                                               # DRAM timing limit (CK)
NCCDAB = tuple(n for n in (4, 5, 6, 7, 8, 10, 12, 16, 24) if n >= N_T) + ((9,) if PRESET == "HBM3_8.0Gbps" else ())
# 9: our choice at 8.0 Gbps (patch 02; AttAcc has no 8.0 Gbps preset)
SEQLENS = (1024, 2048, 4096)
DHEAD, NHEAD, DBYTE = 128, 64, 2       # LLaMA-65B head size (8192 / 64), 64 heads per HBM stack (AttAcc fast mode)

YAML = """Frontend:
  impl: PIMLoadStoreTrace
  path: {trace}
  clock_ratio: 1
  Translation:
    impl: NoTranslation
    max_addr: 2147483648
MemorySystem:
  impl: PIMDRAM
  clock_ratio: 1
  DRAM:
    impl: HBM3-PIM
    org:
      preset: HBM3_8Gb_2R
      channel: 16
    timing:
      preset: {preset}
      nCCDAB: {nccdab}
      nRFC: {nrfc}
  Controller:
    impl: HBM3-PIM
    Scheduler:
      impl: PIM
    RefreshManager:
      impl: AllBankHBM3
    plugins:
  AddrMapper:
    impl: HBM3-PIM
"""


def run(args):
    seqlen, nccdab, tmp = args
    trace = os.path.join(tmp, f"t_{seqlen}.trace")
    yml = os.path.join(tmp, f"c_{seqlen}_{nccdab}.yaml")
    open(yml, "w").write(YAML.format(trace=trace, nccdab=nccdab, preset=PRESET, nrfc=RS.cycles(RS.MAIN_TRFC[PRESET], PRESET)))
    out = subprocess.run([os.path.join(RAM, "ramulator2"), "-f", yml], capture_output=True, text=True, check=True).stdout
    cyc = mac = None
    for line in out.splitlines():
        if "memory_system_cycles" in line:
            cyc = int(line.split()[-1])
        elif "total_num_pim_mac_all_bank_requests" in line:
            mac = int(line.split()[-1])
    return dict(seqlen=seqlen, nccdab=nccdab, cycles=cyc, macab=mac)


def main():
    C.need(C.RAMULATOR, "attacc")
    with tempfile.TemporaryDirectory() as tmp:
        for L in SEQLENS:
            subprocess.run([sys.executable, os.path.join(RAM, "trace_gen", "gen_trace_attacc_bank.py"), "--dhead", str(DHEAD),
                            "--nhead", str(NHEAD), "--seqlen", str(L), "--dbyte", str(DBYTE),
                            "--output", os.path.join(tmp, f"t_{L}.trace")], check=True, capture_output=True)
        with ThreadPoolExecutor(max_workers=6) as ex:
            rows = list(ex.map(run, [(L, n, tmp) for L in SEQLENS for n in NCCDAB]))
    base = {L: next(r["cycles"] for r in rows if r["seqlen"] == L and r["nccdab"] == N_T) for L in SEQLENS}
    for r in rows:
        r["mac_rate_rel"] = N_T / r["nccdab"]               # PIM MAC power relative to no-constraint (timing limit)
        r["attn_speed_rel"] = base[r["seqlen"]] / r["cycles"]
    with open(C.out(f"mac_sweep_{PRESET}.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    for r in rows:
        print(f"L={r['seqlen']:5d} nCCDAB={r['nccdab']:2d}  MAC rate {r['mac_rate_rel']:.3f}  cycles {r['cycles']:7d}  "
              f"attention speed {r['attn_speed_rel']:.3f}")


if __name__ == "__main__":
    main()
