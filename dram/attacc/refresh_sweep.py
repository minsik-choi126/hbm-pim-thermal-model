"""PIM attention time vs refresh rate, from the AttAcc (ASPLOS'24) bank-level Ramulator2 model (temperature -> speed link).

Above 85 degC the refresh rate is doubled (tREFI halved): an assumption from DDR4 practice; JESD238B.01 Table 44 gives
the 1x / 0.5x / 0.25x tREFI rates and leaves the trip temperatures to the vendor. All-bank refresh (REFab) closes every bank of a pseudo
channel for tRFC, and an all-bank PIM MAC needs every bank of the channel open, so PIM stalls for that time. This runs
the same attention trace as mac_sweep.py with no refresh, normal refresh (preset nREFI = 3.9 us) and doubled
refresh (nREFI / 2) at the H100 (HBM3 5.2 Gbps), H200 (6.4 Gbps) and B200 (HBM3E 8.0 Gbps, preset from
dram/patches/attacc/02) data rates.

tRFC follows the channel density, as the simulator's tRFC table does (HBM densities are per channel; the simulator
cites JESD238A): H100's 16 GB stack has 8 Gb per channel -> 350 ns, the 24 GB stacks of H200/B200 have 12 Gb per
channel -> 410 ns (JESD238B.01 Table 93, 16 Gb and 24 Gb die, 8-High). Reading AttAcc's table by die density instead
(450 / 610 ns) is run as a sensitivity case; 610 ns is AttAcc's table value and is not in JESD238B.01 (largest tRFCab
450 ns). Every run uses AttAcc's organisation HBM3_8Gb_2R with nRFC set explicitly. JESD238B.01 Table 4 gives the
same banks, page and columns for 16 Gb and 24 Gb dies (only the row count differs), so the MAC rate does not depend on
it, and AttAcc has no 12 Gb/channel 2-rank preset. AttAcc's trace generator also assumes its 2^14 rows: a larger
organisation shifts the address bits so that only half the channels are used.

Per-bank refresh (PerBankHBM3, dram/patches/attacc/01): every bank gets one REFsb (the simulator's name for JESD238's
per-bank REFpb) per tREFI. With AttAcc's all-bank MAC every bank must be open, so the channel is closed (PREA), the banks are refreshed and ACTAB re-opens them after
tRFCpb. "batched" groups the REFsb of a tREFI back to back, alternating SIDs (a PIM-aware controller; the usual
"spread" schedule, one bank every tREFI / 32, never lets ACTAB through when tRFCpb > tREFI / 32 and is checked
separately in refresh_cmd_check.py). Two timing sets:
  pbj  (main)        JESD238B.01 timings: tRFCpb from Table 93 by die density (200 ns for the 16 Gb die of H100,
                     240 ns for the 24 Gb die of H200/B200) and tRREFD 8 ns (preset nRREFD) counted per pseudo
                     channel, as Table 3 counts it.
  pbh  (comparison)  the HBM2 model of Ramulator 2 (src/dram/impl/HBM2.cpp:19 @b7c7027, nRFCSB 160 at tCK 1 ns,
                     i.e. 160 ns at tRFC 350 ns) scaled with tRFC (160 / 187 ns), tRREFD counted per SID as upstream
                     Ramulator 2 HBM3 does. Shorter than JESD238 and therefore optimistic for per-bank refresh.
With JESD238 timings batched per-bank refresh is slower than all-bank refresh at 2x (pbj2x vs 2x). The controller lets
PIM commands that need no precharge finish the open row before a refresh (explicit policy). The attention trace is
repeated REPEAT times so the run spans many tREFI (steady state; REPEAT 4-16 move the 2x/1x ratio by at most 0.4 points for
all-bank and 0.5 for per-bank refresh, 8 against 16 by 0.2 / 0.3: dram/checks/repeat_check.py).

python -m dram.attacc.refresh_sweep      # writes dram/results/refresh_sweep.csv
"""
import csv
import os
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor

from dram import common as C

RAM = C.RAM
PRESETS = {"HBM3_5.2Gbps": 5070, "HBM3_6.4Gbps": 6240, "HBM3_8.0Gbps": 7800}   # preset nREFI (cycles) = 3.9 us
RATE = {"HBM3_5.2Gbps": 5200, "HBM3_6.4Gbps": 6400, "HBM3_8.0Gbps": 8000}
ORG = "HBM3_8Gb_2R"                                              # AttAcc's organisation (16 GB stack, 2^14 rows)
# all-bank tRFC (ns) per data rate: (channel density = main, die density = sensitivity; 610 = AttAcc table, not JESD238B.01)
TRFC = {"HBM3_5.2Gbps": (350, 450), "HBM3_6.4Gbps": (410, 610), "HBM3_8.0Gbps": (410, 610)}
MAIN_TRFC = {p: t[0] for p, t in TRFC.items()}
RUNS = [(p, t) for p, ts in TRFC.items() for t in ts]


def n_timing(preset):
    """DRAM timing limit of the all-bank MAC interval (CK): the same-bank-group column delay tCCDL = max(4 nCK, 2.5 ns)
    (JESD238B.01 Table 93), with tCK = 4 / data rate (4 transfers per CK): 4 CK at 5.2 and 6.4 Gbps, 5 CK at 8.0 Gbps."""
    return max(4, -(-2500 * RATE[preset] // 4_000_000))       # ceil(2.5 ns / tCK), tCK = 4e6 / rate ps (rate in Mb/s)


N_TIMING = {p: n_timing(p) for p in RATE}
NCCDAB = (4, 6, 7, 8, 12, 16)
NCCDAB_OF = {"HBM3_8.0Gbps": (5, 6, 7, 8, 9, 12, 16)}          # 5: timing limit, 9: our choice at 8.0 Gbps (patch 02;
                                                               # AttAcc has no 8.0 Gbps preset)
SEQLEN = 2048
DHEAD, NHEAD, DBYTE = 128, 64, 2       # as mac_sweep.py (LLaMA-65B head size 8192 / 64, FP16)

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
{dramopt}    org:
      preset: {org}
      channel: 16
    timing:
      preset: {preset}
      nCCDAB: {nccdab}
{refi}{rfc}{rfcsb}  Controller:
    impl: HBM3-PIM
    Scheduler:
      impl: PIM
    RefreshManager:
      impl: {refresh}
{policy}    plugins:
  AddrMapper:
    impl: HBM3-PIM
"""


MODES = ("none", "1x", "2x", "pbj1x", "pbj2x", "pbh1x", "pbh2x")
# pbj: per-bank batched, JESD238B.01 tRFCpb and tRREFD per pseudo channel (main per-bank mode)
# pbh: per-bank batched, HBM2-derived tRFCsb scaled with tRFC and tRREFD per SID (comparison)
REPEAT = 8
# JESD238B.01 Table 93 tRFCpb (ns) by die density: H100 16 GB stack = 16 Gb die, H200/B200 24 GB stack = 24 Gb die
TRFCPB = {"HBM3_5.2Gbps": 200, "HBM3_6.4Gbps": 240, "HBM3_8.0Gbps": 240}


def trfcsb_ns(preset, trfc, mode):
    """Per-bank refresh time (ns) of a per-bank mode: JESD238 tRFCpb (pbj) or the HBM2 model's 160 ns scaled with
    tRFC (pbh)."""
    if mode.startswith("pbj"):
        return TRFCPB[preset]
    return round(160 * trfc / 350)


def cycles(ns, preset):
    return -(-ns * 1000 // int(1e6 / (RATE[preset] / 4)))


def config(preset, trfc, nccdab, mode, trace, plugin=""):
    """Ramulator2 YAML for one run (organisation ORG, all-bank tRFC `trfc` ns)."""
    refi = f"      nREFI: {PRESETS[preset] // 2}\n" if mode.endswith("2x") else ""
    rfc = f"      nRFC: {cycles(trfc, preset)}\n"
    per_bank = mode.startswith("pb")
    rfcsb = f"      nRFCSB: {cycles(trfcsb_ns(preset, trfc, mode), preset)}\n" if per_bank else ""
    policy = "      policy: batched\n" if per_bank else ""
    refresh = "No" if mode == "none" else ("PerBankHBM3" if per_bank else "AllBankHBM3")
    dramopt = "    refsb_rrefd_level: pseudochannel\n" if mode.startswith("pbj") else ""
    return YAML.format(trace=trace, org=ORG, preset=preset, nccdab=nccdab, refi=refi, rfc=rfc, rfcsb=rfcsb,
                       refresh=refresh, policy=policy, dramopt=dramopt).replace("    plugins:\n", "    plugins:\n" + plugin)


def run(args):
    preset, trfc, nccdab, mode, trace, tmp = args
    yml = os.path.join(tmp, f"c_{preset}_{trfc}_{nccdab}_{mode}.yaml")
    open(yml, "w").write(config(preset, trfc, nccdab, mode, trace))
    out = subprocess.run([os.path.join(RAM, "ramulator2"), "-f", yml], capture_output=True, text=True, check=True).stdout
    cyc = mac = None
    for line in out.splitlines():
        if "memory_system_cycles" in line:
            cyc = int(line.split()[-1])
        elif "total_num_pim_mac_all_bank_requests" in line:
            mac = int(line.split()[-1])
    return dict(preset=preset, trfc_ns=trfc, nccdab=nccdab, refresh=mode, cycles=cyc, macab=mac)


def make_trace(path):
    """AttAcc bank-level attention trace, repeated REPEAT times."""
    one = path + ".one"
    subprocess.run([sys.executable, os.path.join(RAM, "trace_gen", "gen_trace_attacc_bank.py"), "--dhead", str(DHEAD),
                    "--nhead", str(NHEAD), "--seqlen", str(SEQLEN), "--dbyte", str(DBYTE), "--output", one],
                   check=True, capture_output=True)
    body = open(one).read()
    if not body.endswith("\n"):
        body += "\n"
    open(path, "w").write(body * REPEAT)


def main():
    C.need(C.RAMULATOR, "attacc")
    with tempfile.TemporaryDirectory() as tmp:
        trace = os.path.join(tmp, "t.trace")
        make_trace(trace)
        jobs = [(p, t, n, m, trace, tmp) for p, t in RUNS for n in NCCDAB_OF.get(p, NCCDAB) for m in MODES]
        with ThreadPoolExecutor(max_workers=8) as ex:
            rows = list(ex.map(run, jobs))
    key = lambda r: (r["preset"], r["trfc_ns"], r["nccdab"])
    base = {key(r): r["cycles"] for r in rows if r["refresh"] == "none"}
    one = {(key(r), r["refresh"]): r["cycles"] for r in rows if r["refresh"].endswith("1x")}
    for r in rows:
        r["time_vs_no_refresh"] = r["cycles"] / base[key(r)]
        same_1x = r["refresh"].replace("2x", "1x") if r["refresh"].endswith("2x") else r["refresh"]
        r["time_vs_1x"] = r["cycles"] / one[(key(r), same_1x)] if same_1x != "none" else r["cycles"] / one[(key(r), "1x")]
    with open(C.out("refresh_sweep.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    for r in rows:
        if r["refresh"].endswith("2x"):
            one_mode = r["refresh"].replace("2x", "1x")
            print(f"{r['preset']:13s} tRFC {r['trfc_ns']} ns nCCDAB {r['nccdab']:2d} {r['refresh']:6s}: 1x +{100 * (one[(key(r), one_mode)] / base[key(r)] - 1):.1f} % "
                  f"| 2x +{100 * (r['time_vs_no_refresh'] - 1):.1f} % vs none, +{100 * (r['time_vs_1x'] - 1):.1f} % vs 1x")


if __name__ == "__main__":
    main()
