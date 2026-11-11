"""IDD7 stack power from public HBM data against AttAcc's PIM power constraint, and PIM energy per byte from the same data.

HBM3 IDD7 values are vendor datasheet values and not public; JESD238B.01 defines the IDD7 loop (Table 89) without values.
The public data are those of the HBM-Power artifact of Olgun et al. (v0.3-alpha, fetched by dram/setup/hbm_power.sh).

IDD7 estimates, W per stack (dynamic):
    HBM2 chips    data/all_idd_measurements.csv (IDD loops measured on 35 HBM2 chips with an IDD7 row): per chip the
                  median over samples of r = (IDD7 - IDD3N1) / (IDD4R - IDD3N1), times the GPU's measured dynamic memory
                  power per stack with a full-duty copy kernel (dram/data/measured_copy_power.json). Lowest, median
                  and highest chip; HBM2 ratios used for HBM3.
    HBM3E model   the JESD238 IDD7 loop (per pseudo channel one ACT per 8 CK, a read every 2 CK) on the artifact's
                  HBM3 6.4 Gb/s configuration (sources/figure16/configs), read current scaled to 5.2 Gb/s, random data
                  (DQ, TSV and bank-group toggle 0.5, as the artifact's figure16.py); dynamic = minus IDD3N1, the
                  background the loop is built on (as DRAMPower's HBM3 model). These values hold for v0.3-alpha; the
                  HBM3E config of HBM-Power main (61aada9) gives 0.96x / 1.22x of AttAcc's PIM power.
Each estimate over AttAcc's PIM power per stack at its power-constrained MAC interval (dram/results/pim_table.json).

PIM energy per byte against AttAcc's table (4.56 pJ/B): one ACT per 1 KB row + the in-bank share 0.44 / 3.18 of a
normal read + MAC 0.16 pJ/B, from the HBM2 chips (currents fitted at 60 degC, median over chips) and from the HBM3E
model (in-bank read energy data-independent up to scaled like a random-data read). The read energy is the IDD4R
current above IDD3N1, as DRAMPower computes it.

The copy kernel reads and writes with activations, so r x copy power counts write and ACT power too and is high for a
read-only PIM stream: the HBM2 estimates are upper-side.

python -m dram.hbm_power.idd7_chips      # writes dram/results/idd7_chips.json
"""
import csv
import json
import os
import statistics as st
import subprocess

import numpy as np

from dram import common as C
from dram.attacc import pim_table as PT

ART = C.HBM_POWER
DATA = os.path.join(ART, "data", "all_idd_measurements.csv")
HBM3_CFG = os.path.join(ART, "sources", "figure16", "configs")
DP_MODEL = os.path.join("sources", "drampower", "src", "DRAMPower", "DRAMPower", "util", "datapattern_model.h")
ART_INPUTS = ("data", os.path.join("sources", "figure16", "configs"), os.path.join("scripts", "figures", "figure16.py"),
              DP_MODEL)
ZENODO = {"v0.3-alpha": "10.5281/zenodo.21970747"}   # version DOI (concept DOI 10.5281/zenodo.21531486)
COPY = os.path.join(C.DATA, "measured_copy_power.json")

ATT_PIM_PJB = 4.56                # AttAcc table: (0.11 + 0.44) x 8 + 0.16 pJ/B
BANK_SHARE = 0.44 / 3.18          # in-bank share of a normal read (AttAcc table: bank read / whole read path). The
                                  # 0.44 / 1.01 / 1.23 split is AttAcc's own (Sec. 7.1, datapath-length scaling), not
                                  # FGDRAM's. 3.18 leaves out ACT and the 0.3 interposer I/O, which errs toward more
                                  # PIM heat (0.44 / 3.48 would give 1.24x instead of 1.35x).
MAC_PJB = PT.MAC_PJ / 2           # 0.32 pJ per FP16 MAC
ROW_B = 1024                      # bytes per row and pseudo channel, one ACT per row read by PIM
HBM2_V, HBM2_TCK = 1.2, 1.6667e-9                     # artifact HBM2 configs (sources/table3/configs)
HBM2_NPC, HBM2_BITS = 8, 256      # 8 x 256 bits per 2 CK: the artifact's HBM2 calibration, K = 512 mA per pJ/bit
                                  # (HBM-Power main docs/explanation/idd-derivation.md:38)
T_REF, T_SPAN_MIN, T_SLOPE = 60.0, 4.0, 6.7   # degC; fallback slope mA/degC for chips with a narrow temperature span
LOOPS = ("IDD0", "IDD2", "IDD3N1", "IDD3N16", "IDD4R", "IDD7")


def artifact_version():
    """Tag and commit of the HBM-Power checkout and whether the files read here differ from that commit."""
    def git(*a):
        try:
            return subprocess.run(["git", "-C", ART, *a], capture_output=True, text=True, check=True).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            return None
    tag = git("describe", "--tags", "--exact-match") or git("describe", "--tags")
    dirty = git("status", "--porcelain", "--", *ART_INPUTS)
    return dict(tag=tag, commit=git("rev-parse", "HEAD"), zenodo_doi=ZENODO.get(tag),
                inputs_unmodified=None if dirty is None else dirty == "", inputs=list(ART_INPUTS))


def load():
    """(chip, loop) -> (currents mA, temperatures degC) for the loops used here."""
    cur = {}
    with open(C.need(DATA, "hbm_power")) as f:
        for r in csv.DictReader(f):
            if r["test_loop"] in LOOPS:
                c = cur.setdefault((r["chip_id"], r["test_loop"]), ([], []))
                c[0].append(float(r["idd"]))
                c[1].append(float(r["temperature"]))
    return cur


def chip_ratios(cur):
    out = []
    for chip in sorted({c for c, _ in cur}, key=int):
        if not all((chip, k) in cur for k in ("IDD7", "IDD4R", "IDD3N1")):
            continue
        i7, i4, i3 = (st.median(cur[(chip, k)][0]) for k in ("IDD7", "IDD4R", "IDD3N1"))
        out.append(dict(chip=int(chip), idd7=i7, idd4r=i4, idd3n=i3, total=i7 / i4, dynamic=(i7 - i3) / (i4 - i3)))
    return out


def at_t_ref(idd, temp):
    """Current at T_REF from a linear fit against temperature; median + T_SLOPE x offset when the span is narrow."""
    a, t = np.array(idd), np.array(temp)
    if np.ptp(t) < T_SPAN_MIN:
        return float(np.median(a) + T_SLOPE * (T_REF - np.median(t))), True
    return float(np.polyval(np.polyfit(t, a, 1), T_REF)), False


def hbm2_pim_energy(cur):
    """PIM energy per byte from the HBM2 chips, median over the chips."""
    need = ("IDD0", "IDD2", "IDD3N1", "IDD4R")
    chips, narrow = [], 0
    for chip in sorted({c for c, _ in cur}, key=int):
        if not all((chip, k) in cur for k in need):
            continue
        m = {}
        for k in need:
            m[k], nw = at_t_ref(*cur[(chip, k)])
            narrow += nw
        bg = (21 * m["IDD3N1"] + 11 * m["IDD2"]) / 32    # IDD0 loop: bank open 21 of 32 CK (our reading of the
                                                         # artifact's IDD0 loop; effect < 0.5 %)
        e_act = HBM2_V * (m["IDD0"] - bg) * 1e-3 * 32 * HBM2_TCK / HBM2_NPC * 1e12              # pJ per ACT, pseudo channel
        e_rd = HBM2_V * (m["IDD4R"] - m["IDD3N1"]) * 1e-3 * 2 * HBM2_TCK / (HBM2_NPC * HBM2_BITS) * 1e12   # pJ/bit
        chips.append(dict(chip=int(chip), act_pJ=e_act, read_pJ_bit=e_rd))
    e_act, e_rd = (float(np.median([c[k] for c in chips])) for k in ("act_pJ", "read_pJ_bit"))
    pim = e_act / ROW_B + BANK_SHARE * e_rd * 8 + MAC_PJB
    return dict(n_chips=len(chips), t_ref_C=T_REF, narrow_span_fits=narrow, act_pJ=e_act, read_pJ_bit=e_rd,
                pim_pJ_B=pim, pim_over_table=pim / ATT_PIM_PJB, chips=chips)


def hbm3_model(rate_gbps=5.2):
    """JESD238 IDD7 loop and PIM energy on the HBM3 configuration of the power model."""
    cfg = json.load(open(os.path.join(HBM3_CFG, "HBM3_6400_power_datapattern.json")))
    tim = json.load(open(os.path.join(HBM3_CFG, "HBM3_6400Mbps_timing.json")))["timing"]
    I, V, dp = cfg["IDD"], cfg["voltage"]["VDD"], cfg["datapattern"]
    assert not dp["use_coupling"] and (dp["ref_dq_rate"], dp["ref_tsv_rate"], dp["ref_bg_rate"]) == (0.0, 0.0, 0.0)
    tck = tim["tCK_ps"] * 1e-12                                       # 6.4 Gbps clock
    # one ACT window, tRAS + tRP, above the open / precharged background (as DRAMPower's HBM3 model)
    e_act = V * (I["IDD0"] * (tim["nRAS"] + tim["nRP"]) - I["IDD3N1"] * tim["nRAS"] - I["IDD2N"] * tim["nRP"]) * 1e-3 * tck
    pj = lambda dq, tsv, bg: dp["floor_pJbit"] + dq * dp["coef_T_DQ"] + tsv * dp["coef_T_2bit"] + bg * dp["coef_busflip"]
    out = dict(rate_gbps=rate_gbps, e_act_nJ=e_act * 1e9, cases={})
    tck_r = 4 / (rate_gbps * 1e9)                                     # HBM3: 4 transfers per CK
    knobs = {"data 0": (0.0, 0.0, 0.0),                               # the configuration's all-zero reference
             "random, config knobs": (dp["dq_rate"], dp["tsv_rate"], dp["bg_rate"]),   # DQ, TSV 0.5, bank group 0
             "random": (0.5, 0.5, 0.5)}                               # the artifact's random data (figure16.py)
    for case, (dq, tsv, bg) in knobs.items():
        scale = pj(dq, tsv, bg) / pj(0, 0, 0)
        rd = (I["IDD4R"] - I["IDD3N1"]) * rate_gbps / 6.4 * scale     # mA per pseudo channel, read bus 100 %
        act = e_act / V / (8 * tck_r) * 1e3                           # mA, one ACT per 8 CK
        idd7 = I["IDD3N1"] + rd + act
        stack_W = 32 * V * (idd7 - I["IDD3N1"]) * 1e-3
        read_pjbit = V * rd * 1e-3 * (2 * tck_r) / 256 * 1e12         # one 32 B read per 2 CK per pseudo channel
        pim_pjB = e_act * 1e12 / ROW_B + BANK_SHARE * read_pjbit * 8 + MAC_PJB
        out["cases"][case] = dict(dq=dq, tsv=tsv, bg=bg, read_scale=scale, idd7_pc_mA=idd7, read_mA=rd, act_mA=act,
                                  stack_W=stack_W, full_read_stack_W=32 * V * rd * 1e-3, read_pJ_bit=read_pjbit,
                                  pim_pJ_B=pim_pjB, pim_over_table=pim_pjB / ATT_PIM_PJB)
    c = out["cases"]
    out["idd7_case"] = "random"
    out["pim_energy_range"] = dict(min=c["data 0"]["pim_over_table"], max=c["random"]["pim_over_table"])
    return out


def main():
    cur = load()
    chips = chip_ratios(cur)
    dyn = sorted(c["dynamic"] for c in chips)
    pt = PT.table()
    copy = json.load(open(COPY))["gpus"]
    h3 = hbm3_model()
    h2e = hbm2_pim_energy(cur)
    gpus = {}
    for g, v in pt["gpus"].items():
        read_dyn = copy[g]["full_copy"] - copy[g]["idle"]             # W per stack, full copy traffic (dynamic)
        ref = v["pim_W_stack"]                                        # AttAcc's PIM power per stack
        lines = {"hbm2_lowest": dyn[0] * read_dyn, "hbm2_median": st.median(dyn) * read_dyn,
                 "hbm2_highest": dyn[-1] * read_dyn}
        if g == "H100":
            lines["hbm3e_model"] = h3["cases"][h3["idd7_case"]]["stack_W"]
        gpus[g] = dict(copy_dynamic_W_stack=read_dyn, attacc_pim_W_stack=ref, idd7_W_stack=lines,
                       idd7_over_attacc={k: w / ref for k, w in lines.items()},
                       range_over_attacc=[min(lines.values()) / ref, max(lines.values()) / ref],
                       chips_allowing_timing=sum(r * read_dyn >= v["pim_W_timing_stack_step_avg"] for r in dyn))
    res = dict(artifact=artifact_version(), source="data/all_idd_measurements.csv", n_chips=len(chips),
               total_ratio_median=st.median(c["total"] for c in chips),
               dynamic_ratio=dict(median=st.median(dyn), min=dyn[0], max=dyn[-1]), gpus=gpus,
               pim_energy=dict(hbm2_over_table=h2e["pim_over_table"], hbm3e_model_min=h3["pim_energy_range"]["min"],
                               hbm3e_model_max=h3["pim_energy_range"]["max"]),
               hbm2_pim_energy=h2e, hbm3_model=h3, chips=chips)
    with open(C.out("idd7_chips.json"), "w") as f:
        json.dump(res, f, indent=1)
    a = res["artifact"]
    print(f"HBM-Power {a['tag']} ({(a['commit'] or '')[:7]}), inputs unmodified: {a['inputs_unmodified']}")
    print(f"{len(chips)} HBM2 chips: dynamic IDD7 / IDD4R ratio median {st.median(dyn):.3f} ({dyn[0]:.3f}-{dyn[-1]:.3f})")
    for g, v in gpus.items():
        print(f"{g}: copy dynamic {v['copy_dynamic_W_stack']:.2f} W/stack, AttAcc PIM {v['attacc_pim_W_stack']:.1f} W; IDD7 "
              + ", ".join(f"{k} {w:.1f} W" for k, w in v["idd7_W_stack"].items())
              + f"; {v['range_over_attacc'][0]:.2f}-{v['range_over_attacc'][1]:.2f}x AttAcc")
    pe = res["pim_energy"]
    print(f"PIM energy over AttAcc's table: HBM2 chips {pe['hbm2_over_table']:.2f}x, HBM3E model "
          f"{pe['hbm3e_model_min']:.2f}-{pe['hbm3e_model_max']:.2f}x")


if __name__ == "__main__":
    main()
