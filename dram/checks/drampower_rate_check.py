"""DRAMPower bank power at the data rate of the trace.

The command trace of the bank power map (bank_power/regenerate) comes from LLMSimulator's Ramulator 2 at HBM3 5.2 Gb/s
(tCK 0.769 ns), but the committed per-bank energies were priced with HBM-Power's 6.4 Gb/s timing file (tCK 625 ps), so
the 105,305-cycle trace spans 65.8 us instead of 81.0 us. This reruns the patched HBM3_runner on both pseudo channels
with (a) the 6.4 Gb/s file, as committed, and (b) a 5.2 Gb/s file: tCK 769 ps and every timing kept in ns
(ceil(n x 625 / 769) cycles, nREFI 5070, tCCD at the JESD238B.01 minima). Currents and data toggle (0.373, bank group 0)
are unchanged; HBM-Power has no 5.2 Gb/s current set.

Needs bash dram/setup/hbm_power.sh --runner.
python -m dram.checks.drampower_rate_check  # output: dram/checks/out/drampower_rate_check.json
"""
import gzip
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile

from dram import common as C

CFG = os.path.join(C.HBM_POWER, "sources", "figure16", "configs")
RUNNER = os.path.join(C.HBM_POWER, "sources", "drampower", "build", "bin", "HBM3_runner")
REGEN = os.path.join(C.DRAM, "bank_power", "regenerate")
LOG = os.path.join(REGEN, "h100_llama8b_decode_bs1_ctx1024_cmd_hbm3.log.ch0.gz")
COMMITTED = os.path.join(C.DRAM, "bank_power", "data", "h100_llama8b_decode_bs1_pc{}_weights.json")
OUT = os.path.join(C.DRAM, "checks", "out", "drampower_rate_check.json")
ARGS = ["--dq-rate=0.373", "--tsv-rate=0.373", "--bg-rate=0"]


def timing_5200(tmp):
    t = json.load(open(os.path.join(CFG, "HBM3_6400Mbps_timing.json")))["timing"]
    n = {k: (v if k in ("nBL", "nCCD_S", "nCCD_L") else math.ceil(v * 625 / 769 - 1e-9)) for k, v in t.items()}
    n.update(tCK_ps=769, nREFI=5070)
    path = os.path.join(tmp, "HBM3_5200Mbps_timing.json")
    json.dump({"timing": n}, open(path, "w"), indent=1)
    return path, n


def totals(path):
    t = json.load(open(path))["totals"]
    return {k: t[k] for k in ("avg_power_mW", "duration_ns", "total_energy_pJ", "refresh_energy_pJ", "shared_sum_pJ")}


def main():
    C.need(RUNNER, "hbm_power")
    with tempfile.TemporaryDirectory() as tmp:
        raw = os.path.join(tmp, "cmd.log.ch0")
        with gzip.open(LOG, "rb") as fi, open(raw, "wb") as fo:
            shutil.copyfileobj(fi, fo)
        prefix = os.path.join(tmp, "trace")
        subprocess.run([sys.executable, os.path.join(REGEN, "convert_hbm3_log.py"), raw, prefix], check=True, capture_output=True)
        t52, n52 = timing_5200(tmp)
        timings = {"6.4 Gb/s file (as committed)": os.path.join(CFG, "HBM3_6400Mbps_timing.json"), "5.2 Gb/s file": t52}
        res = {}
        for name, tim in timings.items():
            pcs = []
            for pc in (0, 1):
                js = os.path.join(tmp, f"r_{pc}.json")
                subprocess.run([RUNNER, os.path.join(CFG, "HBM3_16Gb_8hi_organization.json"), tim,
                                os.path.join(CFG, "HBM3_6400_power_datapattern.json"), f"{prefix}_ch0_pc{pc}.csv", *ARGS,
                                f"--json={js}"], check=True, capture_output=True)
                pcs.append(totals(js))
            res[name] = dict(pseudo_channels=pcs, stack_W=sum(p["avg_power_mW"] for p in pcs) / 2 * 32 / 1000)
    com = [totals(COMMITTED.format(pc)) for pc in (0, 1)]
    res["committed files"] = dict(pseudo_channels=com, stack_W=sum(p["avg_power_mW"] for p in com) / 2 * 32 / 1000)
    a, b = res["6.4 Gb/s file (as committed)"]["stack_W"], res["5.2 Gb/s file"]["stack_W"]
    out = dict(note=__doc__.split("\n\n")[1].replace("\n", " "), timing_5200=n52, runs=res,
               stack_W_change_5200_vs_6400_pct=100 * (b / a - 1),
               stack_W_change_5200_vs_committed_pct=100 * (b / res["committed files"]["stack_W"] - 1))
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(out, open(OUT, "w"), indent=1)
    for k, v in res.items():
        print(f"{k:30s} stack {v['stack_W']:.4f} W, trace span {v['pseudo_channels'][0]['duration_ns'] / 1000:.1f} us")
    print(f"5.2 Gb/s timing vs 6.4 Gb/s file: {out['stack_W_change_5200_vs_6400_pct']:+.2f} %; "
          f"vs committed: {out['stack_W_change_5200_vs_committed_pct']:+.2f} %")


if __name__ == "__main__":
    main()
