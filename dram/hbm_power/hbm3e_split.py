"""HBM3E read energy split (HBM-Power model of Olgun et al.) and the base-die share x of the in-stack read power.

x here is our I/O-and-TSV lower bound on the HBM-Power HBM3E model, not a published value (HBM-PIM Fig. 11 reads about
49 % for the HBM2 base die and external I/O). It is a prior for a sensitivity case only; the thermal fit sets x.

Configuration: HBM-Power main branch config/HBM3E_6400MTs/power.json (6.4 Gb/s, VDD 1.1 V), with the v0.3-alpha
HBM3 timing and organisation files (sources/figure16/configs); DRAMPower HBM3 equations (rho = 1), random data
(DQ, TSV and bank-group toggle 0.5).
    activation   IDD0 over tRAS + tRP above IDD3N1 (open) / IDD2N (precharged), spread over the reads per activation
                 (whole_row: 32 reads of 32 B per 1 KB row; four_reads: 4, as the artifact's random-read trace)
    read         (IDD4R - IDD3N1) per bit, scaled by the data-pattern model and split by its terms into the
                 data-independent floor, the bank-group bus, the TSVs and the DQ I/O
    in-stack     activation + floor + bank-group bus + TSV + DQ + standby (banks open above precharged)
Base-die share of the in-stack read power: the DQ I/O sits in the base die, the TSVs none, half or all of it, the DRAM
core and standby in the DRAM dies: x_lo = DQ / in-stack, x_mid = (DQ + TSV / 2) / in-stack, x_hi = (DQ + TSV) / in-stack.
The thermal model uses x_mid of whole_row as its prior and the span over both patterns as the prior range.
Checks: Olgun et al.'s H200 measurements (data/HBM3E_measurements.csv, memory power above idle per bit read) and, when
built (bash dram/setup/hbm_power.sh --runner), the artifact's HBM3_runner on its random-read trace.

python -m dram.hbm_power.hbm3e_split      # writes dram/results/hbm3e_split.json
"""
import csv
import json
import os
import re
import subprocess

import numpy as np

from dram import common as C

ART = C.HBM_POWER
HBMP_CFG = os.path.join(ART, "main_config_HBM3E_6400MTs_power.json")
HBMP_TIM = os.path.join(ART, "sources", "figure16", "configs", "HBM3_6400Mbps_timing.json")
HBMP_ORG = os.path.join(ART, "sources", "figure16", "configs", "HBM3_16Gb_8hi_organization.json")
HBMP_MEAS = os.path.join(ART, "data", "HBM3E_measurements.csv")
RUNNER = os.path.join(ART, "sources", "drampower", "build", "bin", "HBM3_runner")
TRACE_RD = os.path.join(ART, "sources", "figure16", "traces", "hbm3_random_read_4rpa.csv")
TRACE_IDLE = os.path.join(ART, "sources", "figure16", "traces", "hbm3_baseline_nop_ref.csv")
N_PC_H200 = 192                                                                 # 6 stacks x 32 pseudo channels
READS_PER_ACT = {"whole_row": 32, "four_reads": 4}                              # 1 KB row / 32 B read; the H200 trace


def hbm3e_split():
    cfg = json.load(open(C.need(HBMP_CFG, "hbm_power")))
    tim, org = json.load(open(HBMP_TIM))["timing"], json.load(open(HBMP_ORG))
    I, V_, dp = cfg["IDD"], cfg["voltage"]["VDD"], cfg["datapattern"]
    tck = tim["tCK_ps"] * 1e-12
    bits = tim["nBL"] * org["width"]                                            # 256 bits per read and pseudo channel
    t_rd = tim["nBL"] / org["dataRate"] * tck                                   # 2 CK
    rate = bits / t_rd                                                          # bit/s per pseudo channel (204.8 Gb/s)
    # DRAMPower HBM3 (core_calculation_HBM3.cpp, rho = 1): E_act = VDD (I_theta - IDD3N1) tRAS, E_pre = 0
    e_act = V_ * (I["IDD0"] * (tim["nRAS"] + tim["nRP"]) - I["IDD2N"] * tim["nRP"] - I["IDD3N1"] * tim["nRAS"]) * 1e-3 * tck
    terms = dict(floor=dp["floor_pJbit"], dq=dp["dq_rate"] * dp["coef_T_DQ"], tsv=dp["tsv_rate"] * dp["coef_T_2bit"],
                 bg_bus=dp["bg_rate"] * dp["coef_busflip"])
    ref = dp["floor_pJbit"] + dp["ref_dq_rate"] * dp["coef_T_DQ"] + dp["ref_tsv_rate"] * dp["coef_T_2bit"] \
        + dp["ref_bg_rate"] * dp["coef_busflip"]
    assert ref == dp["floor_pJbit"]                                             # all-zero reference
    scale = sum(terms.values()) / ref
    e_rd_bit = V_ * (I["IDD4R"] - I["IDD3N1"]) * scale * 1e-3 * t_rd / bits      # J/bit, random data
    part = {k: e_rd_bit * v / sum(terms.values()) * 1e12 for k, v in terms.items()}   # pJ/bit
    standby = V_ * (I["IDD3N1"] - I["IDD2N"]) * 1e-3 / rate * 1e12             # banks open above precharged, at full rate
    pat = {}
    for name, r in READS_PER_ACT.items():
        act = e_act / (r * bits) * 1e12
        core = act + part["floor"] + part["bg_bus"]
        stack = core + part["tsv"] + part["dq"] + standby
        pat[name] = dict(reads_per_act=r, act=act, floor=part["floor"], bg_bus=part["bg_bus"], tsv=part["tsv"],
                         io_dq=part["dq"], standby=standby, dram_core=core, in_stack=stack,
                         core_share=core / stack, tsv_share=part["tsv"] / stack, io_share=part["dq"] / stack,
                         x_lo=part["dq"] / stack, x_mid=(part["dq"] + 0.5 * part["tsv"]) / stack,
                         x_hi=(part["dq"] + part["tsv"]) / stack)
    xs = [p[b] for p in pat.values() for b in ("x_lo", "x_hi")]
    out = dict(config="HBM-Power main 61aada9 config/HBM3E_6400MTs/power.json",
               timing="HBM-Power v0.3-alpha sources/figure16/configs/HBM3_6400Mbps_timing.json", VDD=V_,
               rate_Gbps=org["width"] * org["dataRate"] / tck / 32 / 1e9, e_act_pJ=e_act * 1e12, read_pJ_bit=e_rd_bit * 1e12,
               read_scale_random=scale, read_terms_pJ_bit=part, read_floor_allzero_pJ_bit=e_rd_bit / scale * 1e12,
               unit="pJ/bit", patterns=pat,
               base_die_share=dict(prior=pat["whole_row"]["x_mid"], range=[min(xs), max(xs)]))
    out["h200_measured"] = h200_measured()
    out["runner_check"] = runner_check(e_act * 1e12, e_rd_bit * 1e12, e_rd_bit / scale * 1e12, out["h200_measured"])
    return out


def h200_measured():
    """Olgun et al.'s three H200 runs (random reads): memory power above idle per bit read."""
    rows = list(csv.DictReader(open(HBMP_MEAS)))
    v = [(float(r["total_power_W"]) - float(r["idle_power_W"])) / (float(r["read_GBs"]) * 8e9) * 1e12 for r in rows]
    tot = [float(r["total_power_W"]) for r in rows]
    return dict(n=len(v), active_pJ_bit=float(np.mean(v)), active_pJ_bit_range=[min(v), max(v)],
                total_W_mean=float(np.mean(tot)), read_GBs_mean=float(np.mean([float(r["read_GBs"]) for r in rows])))


def run_runner(trace, knob):
    if not os.path.exists(RUNNER):
        return None
    cmd = [RUNNER, HBMP_ORG, HBMP_TIM, HBMP_CFG, trace, f"--dq-rate={knob}", f"--tsv-rate={knob}", f"--bg-rate={knob}"]
    o = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    txt = o.stdout + o.stderr
    banks = re.findall(r"Bank\s+(\d+) : ACT: (\S+) PRE: \S+ BG_ACT: \S+ BG_PRE: \S+ RD: (\S+) ", txt)
    return dict(act=sum(float(b[1]) for b in banks), rd=sum(float(b[2]) for b in banks),
                power_mW=float(re.search(r"Average power:\s+([\d.]+)\s+mW", txt).group(1)),
                duration_ns=float(re.search(r"duration:\s+([\d.]+)\s+ns", txt).group(1)))


def runner_check(e_act, e_rd, e_rd0, meas):
    """The artifact's HBM3_runner on its random-read trace (4 reads per activation) with the main config: activation
    and read energy equal the equations above; device power above idle per bit against the H200 measurement."""
    a = run_runner(TRACE_RD, 0.5)
    if a is None:
        return dict(ran=False)
    z, idle = run_runner(TRACE_RD, 0.0), run_runner(TRACE_IDLE, 0.0)
    cmds = [r["command"] for r in csv.DictReader(open(TRACE_RD))]
    n_act, n_rd = cmds.count("ACT"), cmds.count("RD")
    ea, er, er0 = a["act"] / n_act, a["rd"] / n_rd / 256, z["rd"] / n_rd / 256
    assert abs(ea / e_act - 1) < 1e-5 and abs(er / e_rd - 1) < 1e-5 and abs(er0 / e_rd0 - 1) < 1e-5, (ea, e_act, er, e_rd)
    bps = n_rd * 256 / (a["duration_ns"] * 1e-9)
    act_bit = (a["power_mW"] - idle["power_mW"]) * 1e-3 / bps * 1e12
    return dict(ran=True, act_pJ=ea, read_pJ_bit=er, read_allzero_pJ_bit=er0, reads_per_act=n_rd / n_act,
                device_total_W=N_PC_H200 * a["power_mW"] / 1e3, device_idle_W=N_PC_H200 * idle["power_mW"] / 1e3,
                device_read_TBs=N_PC_H200 * bps / 8 / 1e12, active_pJ_bit=act_bit,
                vs_measured_active=act_bit / meas["active_pJ_bit"] - 1,
                vs_measured_total=N_PC_H200 * a["power_mW"] / 1e3 / meas["total_W_mean"] - 1)


def main():
    s = hbm3e_split()
    with open(C.out("hbm3e_split.json"), "w") as f:
        json.dump(s, f, indent=1)
    print(f"HBM3E 6.4 Gb/s: ACT {s['e_act_pJ']:.0f} pJ, read {s['read_pJ_bit']:.3f} pJ/bit (random data)")
    for k, p in s["patterns"].items():
        print(f"  {k:10s} in-stack {p['in_stack']:.3f} pJ/bit: core {p['dram_core']:.3f}, TSV {p['tsv']:.3f}, DQ {p['io_dq']:.3f}"
              f"  base-die share x {p['x_lo']:.3f} / {p['x_mid']:.3f} / {p['x_hi']:.3f}")
    b = s["base_die_share"]
    m = s["h200_measured"]
    print(f"prior x {b['prior']:.3f} (range {b['range'][0]:.3f}-{b['range'][1]:.3f}); H200 measured {m['active_pJ_bit']:.2f} pJ/bit "
          f"above idle; runner check: {s['runner_check'].get('ran')}")


if __name__ == "__main__":
    main()
