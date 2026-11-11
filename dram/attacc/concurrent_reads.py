"""PIM + GPU concurrently in AttAcc's bank-level PIM model: how much of the GPU's weight reads overlaps the PIM attention.

A throughput model with ideal overlap assumes: while PIM runs, the GPU keeps its full read bandwidth (u = 1)
and PIM keeps its speed. Here the AttAcc Ramulator2 model runs the attention trace (L = 2K, as mac_sweep.py) and a
fixed amount of ordinary reads together (frontend PIMDualTrace, dram/patches/attacc/03). The reads stand for FC weight
reads: rows apart from the KV rows, spread over every channel, pseudo channel, SID, bank group and bank, each row read
column by column. Three memory models:
    attacc       AttAcc as published: one row buffer per bank, PIM requests before reads. A read to another row closes
                 the row an all-bank MAC needs.
    (reads before PIM requests, controller option pim_priority false, does not finish in AttAcc's controller: even a
    few hundred reads stall it, so it is not run.)
    dual         a second row buffer per bank for PIM (NeuPIMs-style dual row buffers, option dual_row_buffer): PIM and
                 reads keep their own open rows and share only the controller's command slots (one row and one column
                 command per channel per cycle), PIM first. Bank-array conflicts between the two are not modelled, so
                 this is the most overlap such hardware can give.
Per case: T_pim (PIM alone), T_rd (the reads alone) and T (both). The reads are sized so that T_rd = ratio x T_pim
(ratio >= 1: FC longer than the attention, the case the concurrent design aims at). An overlap factor u gives
T = T_rd + (1 - u) T_pim for ratio >= 1, so u = 1 - (T - T_rd) / T_pim (1: ideal overlap, 0: serial). No refresh.

python -m dram.attacc.concurrent_reads      # writes dram/results/concurrent_reads.csv
"""
import csv
import os
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor

from dram import common as C
from dram.attacc import refresh_sweep as RS

# per data rate: AttAcc's MAC interval and the DRAM timing limit (4 / 4 / 5 CK, refresh_sweep.N_TIMING)
CASES = [(pr, n) for pr, n0 in (("HBM3_5.2Gbps", 6), ("HBM3_6.4Gbps", 7), ("HBM3_8.0Gbps", 9)) for n in (n0, RS.N_TIMING[pr])]
MODES = ("attacc", "dual")
RATIOS = (1.0, 2.0)                      # T_rd / T_pim
REPEAT = 1
READ_ALONE = 200_000
W_ROW0, W_ROWS = 12288, 16               # weight rows (KV rows of the trace are 0-24 and 8192-8216)
N_CH, N_PCH, N_RA, N_BG, N_BA, N_COL = 16, 2, 2, 4, 4, 32

YAML = """Frontend:
  impl: PIMDualTrace
  path: "{pim}"
  read_path: {reads}
  read_rate: {rate}
  read_count: {count}
  clock_ratio: 1
  Translation:
    impl: NoTranslation
    max_addr: 2147483648
MemorySystem:
  impl: PIMDRAM
  clock_ratio: 1
  dual_row_buffer: {dual}
  DRAM:
    impl: HBM3-PIM
    org:
      preset: {org}
      channel: 16
    timing:
      preset: {preset}
      nCCDAB: {nccdab}
      nRFC: {nrfc}
  Controller:
    impl: HBM3-PIM
    pim_priority: {pim_first}
    Scheduler:
      impl: PIM
    RefreshManager:
      impl: No
    plugins:
  AddrMapper:
    impl: HBM3-PIM
"""


def addr(ch, pch, ra, bg, ba, row, col):
    """Byte address in AttAcc's layout (low to high: 32 B, column 5, row 14, bank 2, BG 2, SID 1, pCH 1, channel 4)."""
    a = ch
    for v, n in ((pch, N_PCH), (ra, N_RA), (bg, N_BG), (ba, N_BA), (row, 1 << 14), (col, N_COL)):
        a = a * n + v
    return a << 5


def make_reads(path):
    """Weight-read stream: for each row and column, one read to every bank of the stack (channel fastest)."""
    with open(path, "w") as f:
        for row in range(W_ROW0, W_ROW0 + W_ROWS):
            for col in range(N_COL):
                for ba in range(N_BA):
                    for bg in range(N_BG):
                        for ra in range(N_RA):
                            for pch in range(N_PCH):
                                for ch in range(N_CH):
                                    f.write(f"{addr(ch, pch, ra, bg, ba, row, col):#x}\n")


def run(job, mode="attacc"):
    preset, nccdab, pim, reads, rate, count, tmp = job
    yml = os.path.join(tmp, f"c_{preset}_{nccdab}_{rate}_{count}_{bool(pim)}_{mode}.yaml")
    open(yml, "w").write(YAML.format(pim=pim, reads=reads, rate=rate, count=count, org=RS.ORG, preset=preset,
                                     nccdab=nccdab, nrfc=RS.cycles(RS.MAIN_TRFC[preset], preset),
                                     dual=str(mode == "dual").lower(), pim_first=str(mode != "read_first").lower()))
    out = subprocess.run([os.path.join(RS.RAM, "ramulator2"), "-f", yml], capture_output=True, text=True, check=True).stdout
    st = {}
    for line in out.splitlines():
        for key in ("memory_system_cycles", "total_num_pim_mac_all_bank_requests", "dual_reads_sent", "dual_pim_sent_cycle"):
            if line.strip().startswith(key + ":"):
                st[key] = int(line.split()[-1])
    return dict(preset=preset, nccdab=nccdab, mode=mode, pim=bool(pim), rate=rate, cycles=st["memory_system_cycles"],
                macab=st.get("total_num_pim_mac_all_bank_requests", 0), reads=st.get("dual_reads_sent", 0))


def main():
    C.need(C.RAMULATOR, "attacc")
    with tempfile.TemporaryDirectory() as tmp:
        trace, reads = os.path.join(tmp, "t.trace"), os.path.join(tmp, "reads.txt")
        RS.REPEAT, rep = REPEAT, RS.REPEAT
        RS.make_trace(trace)
        RS.REPEAT = rep
        make_reads(reads)
        with ThreadPoolExecutor(max_workers=8) as ex:
            pim = dict(zip(CASES, ex.map(lambda c: run((*c, trace, reads, 0.0, 0, tmp)), CASES)))
            rate = dict(zip(CASES, ex.map(lambda c: run((*c, "", reads, -1.0, READ_ALONE, tmp)), CASES)))
            n_rd = {(c, r): int(r * pim[c]["cycles"] * rate[c]["reads"] / rate[c]["cycles"]) for c in CASES for r in RATIOS}
            rd = dict(zip(n_rd, ex.map(lambda k: run((*k[0], "", reads, -1.0, n_rd[k], tmp)), n_rd)))
            jobs = [(c, r, m) for c in CASES for r in RATIOS for m in MODES]
            both = dict(zip(jobs, ex.map(lambda j: run((*j[0], trace, reads, -1.0, n_rd[(j[0], j[1])], tmp), j[2]), jobs)))
    rows = []
    for (c, r, m), b in both.items():
        tp, tr, t = pim[c]["cycles"], rd[(c, r)]["cycles"], b["cycles"]
        rows.append(dict(preset=c[0], nccdab=c[1], mode=m, ratio=r, t_pim=tp, t_rd=tr, t_both=t,
                         serial=tp + tr, ideal=max(tp, tr), u=1 - (t - tr) / tp,
                         read_TBs_alone=rate[c]["reads"] * 32 / (rate[c]["cycles"] * 1e-6 / (RS.RATE[c[0]] / 4)) / 1e12))
    with open(C.out("concurrent_reads.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    for x in rows:
        print(f"{x['preset']:13s} n{x['nccdab']:<2d} {x['mode']:10s} ratio {x['ratio']:.0f}: PIM {x['t_pim']} reads {x['t_rd']} "
              f"both {x['t_both']} (serial {x['serial']}, ideal {x['ideal']})  u {x['u']:.2f}")


if __name__ == "__main__":
    main()
