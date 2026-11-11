"""Command-level check of the refresh runs in refresh_sweep.py (all-bank and per-bank, AttAcc all-bank PIM).

Runs the sweep's configurations with Ramulator2's HBM3TraceRecorder and checks every issued command against the rules
the refresh results rely on, per channel:
  R1  no ACTAB within tRFCsb after a REFsb (or within tRFC after a REFab) in the channel: a refreshing bank is not re-opened
  R2  no MACAB while the channel is closed: after a PREA, the next all-bank PIM command is an ACTAB
  R3  PREA -> REFsb >= tRP;  PREA -> REFab >= tRP
  R4  REFsb -> REFsb >= tRREFD, in the same pseudo channel (pbj, JESD238B.01 Table 3) or the same SID (pbh)
  R5  every bank (REFsb) / pseudo channel (REFab) is refreshed once per tREFI: refresh count matches the triggers
      before the end of the run, and no refresh is issued later than one tREFI after its trigger (no starvation)
  R6  MACAB -> PREA >= tRTPL, ACTAB -> PREA >= tRAS, MACAB -> MACAB >= nCCDAB, ACTAB -> MACAB >= tRCD (existing rules)
  R7  the MACAB count in the command log equals the run's statistics
  R8  REFsb to the same bank are at least tRFCsb apart
  R9  every one of the 16 channels carries PIM commands (the trace's address mapping matches the organisation)
Also checks that the "spread" per-bank schedule (one bank every tREFI / 32) is refused when tRFCsb > tREFI / 32
(the all-bank PIM could never re-open the channel).

python -m dram.attacc.refresh_cmd_check      # exit status 1 on any violation
"""
import glob
import os
import subprocess
import sys
import tempfile
from collections import defaultdict

from dram import common as C
from dram.attacc import refresh_sweep as SW

# preset timings (cycles) used by the rules; nRFC / nRFCSB / nREFI follow the sweep's overrides below
T = {"HBM3_5.2Gbps": dict(nRP=19, nRAS=45, nRTPL=8, nRREFD=11, nRFCSB=260, nREFI=5070, nRCDRD=19),
     "HBM3_6.4Gbps": dict(nRP=23, nRAS=55, nRTPL=10, nRREFD=13, nRFCSB=320, nREFI=6240, nRCDRD=23),
     "HBM3_8.0Gbps": dict(nRP=28, nRAS=68, nRTPL=12, nRREFD=16, nRFCSB=400, nREFI=7800, nRCDRD=28)}
MODES = ("1x", "2x", "pbj1x", "pbj2x", "pbh2x")
# each data rate with its main tRFC, at the DRAM timing limit (4 / 4 / 5 CK, SW.N_TIMING) and AttAcc's interval
CASES = [(p, SW.MAIN_TRFC[p], n, m) for p, n0 in (("HBM3_5.2Gbps", 6), ("HBM3_6.4Gbps", 7), ("HBM3_8.0Gbps", 9))
         for n in (SW.N_TIMING[p], n0) for m in MODES]
BANKS_PER_PCH, N_PCH = 32, 2       # HBM3 8-high 2R: 2 SIDs x 4 BGs x 4 banks per pseudo channel


def timings(preset, trfc, nccdab, mode):
    t = dict(T[preset], nCCDAB=nccdab)
    t["nRFC"] = SW.cycles(trfc, preset)                 # the sweep sets nRFC explicitly (ceil)
    if mode.startswith("pb"):
        t["nRFCSB"] = SW.cycles(SW.trfcsb_ns(preset, trfc, mode), preset)
    if mode.endswith("2x"):
        t["nREFI"] = SW.PRESETS[preset] // 2
    return t


def parse(path):
    cmds = []
    for line in open(path):
        f = [x.strip() for x in line.split(",")]
        cmds.append((int(f[0]), f[1], tuple(int(x) for x in f[2:])))
    return cmds


def check_channel(cmds, t, mode, end_clk):
    err = []
    per_bank = mode.startswith("pb")
    last = {}                                   # last issue time of a command in the channel
    last_ref = None                             # (time, latency) of the latest refresh
    closed = False                              # channel closed by PREA and not yet re-opened by ACTAB
    last_ref_sid = {}
    refs = defaultdict(list)
    n_mac = 0
    for clk, c, a in cmds:
        if c == "ACTAB":
            if last_ref and clk - last_ref[0] < last_ref[1]:
                err.append(f"R1 ACTAB {clk - last_ref[0]} cycles after refresh (< {last_ref[1]})")
            if "PREA" in last and clk - last["PREA"] < t["nRP"]:
                err.append(f"R3 ACTAB {clk - last['PREA']} after PREA")
            closed = False
        elif c == "MACAB":
            n_mac += 1
            if "ACTAB" in last and clk - last["ACTAB"] < t["nRCDRD"]:
                err.append(f"R6 ACTAB->MACAB {clk - last['ACTAB']}")
            if closed:
                err.append(f"R2 MACAB at {clk} while the channel is closed")
            if "MACAB" in last and clk - last["MACAB"] < t["nCCDAB"]:
                err.append(f"R6 MACAB->MACAB {clk - last['MACAB']}")
        elif c == "PREA":
            if "MACAB" in last and clk - last["MACAB"] < t["nRTPL"]:
                err.append(f"R6 MACAB->PREA {clk - last['MACAB']}")
            if "ACTAB" in last and clk - last["ACTAB"] < t["nRAS"]:
                err.append(f"R6 ACTAB->PREA {clk - last['ACTAB']}")
            closed = True
        elif c in ("REFsb", "REFab"):
            if "PREA" in last and clk - last["PREA"] < t["nRP"]:
                err.append(f"R3 {c} {clk - last['PREA']} after PREA")
            if c == "REFsb":
                pch, sid, bg, bank = a[1], a[2], a[3], a[4]
                unit = (pch,) if mode.startswith("pbj") else (pch, sid)      # tRREFD counting level
                if unit in last_ref_sid and clk - last_ref_sid[unit] < t["nRREFD"]:
                    err.append(f"R4 REFsb->REFsb {clk - last_ref_sid[unit]} (< tRREFD) in {unit}")
                last_ref_sid[unit] = clk
                if refs[(pch, sid, bg, bank)] and clk - refs[(pch, sid, bg, bank)][-1] < t["nRFCSB"]:
                    err.append(f"R8 REFsb same bank {clk - refs[(pch, sid, bg, bank)][-1]}")
                refs[(pch, sid, bg, bank)].append(clk)
                lat = t["nRFCSB"]
            else:
                refs[(a[1],)].append(clk)
                lat = t["nRFC"]
            if last_ref is None or clk + lat > last_ref[0] + last_ref[1]:
                last_ref = (clk, lat)
        last[c] = clk
    # R5: refresh count and lateness
    interval = t["nREFI"]
    triggers = (end_clk - 1) // interval
    units = BANKS_PER_PCH * N_PCH if per_bank else N_PCH
    if len(refs) != units:
        err.append(f"R5 refreshed units {len(refs)} != {units}")
    worst_late = 0
    for u, ts in refs.items():
        if not (triggers - 1 <= len(ts) <= triggers):
            err.append(f"R5 unit {u}: {len(ts)} refreshes, {triggers} triggers")
        for k, clk in enumerate(ts):
            late = clk - (k + 1) * interval
            worst_late = max(worst_late, late)
            if late < 0 or late > interval:
                err.append(f"R5 unit {u} refresh {k} issued {late} cycles after its trigger")
    return err, n_mac, worst_late


def run_case(preset, trfc, nccdab, mode, trace, tmp):
    log = os.path.join(tmp, f"log_{preset}_{trfc}_{nccdab}_{mode}", "cmd")
    plugin = f"    - ControllerPlugin:\n        impl: HBM3TraceRecorder\n        path: {log}\n"
    yml = os.path.join(tmp, f"k_{preset}_{trfc}_{nccdab}_{mode}.yaml")
    open(yml, "w").write(SW.config(preset, trfc, nccdab, mode, trace, plugin))
    out = subprocess.run([os.path.join(SW.RAM, "ramulator2"), "-f", yml], capture_output=True, text=True, check=True).stdout
    cyc = int(next(l for l in out.splitlines() if "memory_system_cycles" in l).split()[-1])
    mac = int(next(l for l in out.splitlines() if "total_num_pim_mac_all_bank_requests" in l).split()[-1])
    t = timings(preset, trfc, nccdab, mode)
    errs, n_mac, late, busy = [], 0, 0, 0
    for f in sorted(glob.glob(log + ".ch*")):
        e, m, lt = check_channel(parse(f), t, mode, cyc)
        errs += [f"{os.path.basename(f)}: {x}" for x in e]
        n_mac, late, busy = n_mac + m, max(late, lt), busy + (m > 0)
    if n_mac != mac:
        errs.append(f"R7 MACAB in log {n_mac} != stats {mac}")
    if busy != 16:
        errs.append(f"R9 PIM commands in {busy} of 16 channels")
    return cyc, late, errs


def main():
    C.need(C.RAMULATOR, "attacc")
    bad = 0
    with tempfile.TemporaryDirectory() as tmp:
        trace = os.path.join(tmp, "t.trace")
        SW.make_trace(trace)
        for case in CASES:
            cyc, late, errs = run_case(*case, trace, tmp)
            bad += len(errs)
            print(f"{case[0]:13s} tRFC {case[1]} ns nCCDAB {case[2]} {case[3]:6s}: cycles {cyc}, worst refresh delay {late} cycles, "
                  f"violations {len(errs)}")
            for e in errs[:5]:
                print("    ", e)
        # spread schedule: must be refused (ACTAB would never get a tRFCsb gap)
        yml = os.path.join(tmp, "spread.yaml")
        open(yml, "w").write(SW.config("HBM3_5.2Gbps", SW.MAIN_TRFC["HBM3_5.2Gbps"], 6, "pbj1x", trace).replace("policy: batched", "policy: spread"))
        r = subprocess.run([os.path.join(SW.RAM, "ramulator2"), "-f", yml], capture_output=True, text=True, timeout=120)
        refused = "starves" in (r.stdout + r.stderr)
        print("spread schedule refused as expected" if refused else "spread schedule NOT refused")
        bad += 0 if refused else 1
    print("TOTAL VIOLATIONS", bad)
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
