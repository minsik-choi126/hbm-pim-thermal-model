"""PIM bandwidth, power, MAC interval, batch and channel load of AttAcc's bank-level PIM on H100, H200 and B200.

Inputs (DRAM_RESULTS if the file is there, else the committed dram/results): mac_sweep_<preset>.csv (AttAcc Ramulator 2, attention trace at context 2048),
refresh_sweep.csv and attacc_e2e_h100_llama65b.csv. Constants: AttAcc's energy table (src/config.py) and timing presets
(HBM3-PIM.cpp), checked against the sources when dram/external/attacc_simulator is there.

    PIM bandwidth per stack   macab x 2048 B / (cycles x tCK), tCK = 4 / data rate (one all-bank MAC command moves
                              2048 B per channel; macab counts all 16 channels of the stack)
    MAC interval              power-constrained nCCDAB: 6 / 7 clocks at 5.2 / 6.4 Gb/s (AttAcc presets); 9 at
                              8.0 Gb/s (our choice, dram/patches/attacc/02) and the DRAM timing limit tCCD_L = max(4 nCK, 2.5 ns)
    PIM energy                (ACT 0.11 + cell read 0.44) pJ/bit x 8 + MAC 0.32 pJ / 2 B = 4.56 pJ/B
    PIM power per stack       PIM energy x PIM bandwidth (every channel at the nominal rate); step average = that /
                              kappa (the most-loaded channel sets the length of the PIM phase)
    batch                     KV-capacity batch: (HBM capacity - weights) // (context x KV bytes per token), LLaMA-65B,
                              TP 8, FP16, context 2048
    channel load (kappa)      (request, KV head) pairs per channel, most loaded / mean: AttAcc places whole pairs,
                              ceil(pairs / stacks) per stack, then over the 16 channels
    PIM attention per layer   KV bytes / (stacks x PIM bandwidth / kappa) / layers, against AttAcc's main.py
    refresh factors           doubled refresh (tREFI 3.9 -> 1.95 us): GPU reads (1 - tRFC/tREFI) / (1 - 2 tRFC/tREFI);
                              PIM attention time 2x / 1x from refresh_sweep.csv (all-bank refresh, AttAcc's interval
                              and the timing limit); per-bank refresh for comparison (r_pim_per_bank: pbj with JESD238
                              timings, pbh with HBM2-derived timings; not used by the thermal model)

python -m dram.attacc.pim_table        # writes dram/results/pim_table.json
"""
import csv
import json
import os
import re

from dram import common as C
from dram.attacc.refresh_sweep import MAIN_TRFC, N_TIMING, RATE

# data rate preset; MAC interval (clocks): AttAcc presets for 5.2 / 6.4 Gb/s, ours for 8.0 Gb/s (patch 02); HBM stacks;
# HBM capacity (B): datasheet decimal GB (conservative; H100's physical 85.9e9 B would give batch 103)
GPUS = {
    "H100": dict(preset="HBM3_5.2Gbps", n_attacc=6, stacks=5, capacity=80e9),
    "H200": dict(preset="HBM3_6.4Gbps", n_attacc=7, stacks=6, capacity=141e9),
    "B200": dict(preset="HBM3_8.0Gbps", n_attacc=9, stacks=8, capacity=180e9),
}
# AttAcc src/config.py: ENERGY_TABLE['PIM'][PIMType.BA]['mem'] = (0.11 + 0.44) x 8 pJ/B (ACT from FGDRAM Table 3, cell
# read AttAcc's own split, Sec. 7.1) and ENERGY_TABLE['PIM'][PIMType.BA]['alu'] = 0.32 pJ per MAC (config.py:58; no cited
# source). The BA 'sram' term (0.0034 pJ/B) is left out. The simulator charges the MAC on the score layer only
# (0.08 pJ/B); we charge K and V (0.16 pJ/B, +1.8 % of the PIM energy).
ACT_PJ_BIT, CELL_PJ_BIT, MAC_PJ = 0.11, 0.44, 0.32
E_BA = (ACT_PJ_BIT + CELL_PJ_BIT) * 8 + MAC_PJ / 2         # pJ per KV byte (FP16: one MAC per 2 B)
SEQLEN, TP = 2048, 8
LAYERS, HIDDEN, HEADS = 80, 8192, 64                       # LLaMA-65B
WEIGHTS = 130e9 / TP                                       # B per GPU: 65.0e9 params x 2 B (LLaMA Table 2 gives
                                                           # 65.2B, which would give B200 batch 243)
KV_TOKEN = 2 * LAYERS * HIDDEN * 2 / TP                    # K and V, FP16, B per context token per GPU
T_REFI = 3.9e-6


def src(name):
    """An input result: the regenerated one in DRAM_RESULTS if present, else the committed one."""
    p = os.path.join(C.RESULTS, name)
    return p if os.path.exists(p) else C.ref(name)


def bw_table(preset):
    """PIM bandwidth per stack (B/s) by MAC interval, context 2048."""
    path = src(f"mac_sweep_{preset}.csv")
    tck = 4 / (RATE[preset] * 1e6)
    with open(path) as f:
        rows = [r for r in csv.DictReader(f) if int(r["seqlen"]) == SEQLEN]
    return {int(r["nccdab"]): int(r["macab"]) * 2048 / (int(r["cycles"]) * tck) for r in rows}


def pairs(batch, stacks):
    """(request, KV head) pairs per channel: mean and most loaded (AttAcc's whole-pair placement)."""
    per_gpu = batch * (HEADS // TP)
    per_stack = -(-per_gpu // stacks)
    return per_gpu / (stacks * 16), -(-per_stack // 16)


def r_pim(preset, n, rows, mode="2x"):
    """PIM attention time at 2x refresh over 1x of the same refresh mode (all-bank "2x", per-bank "pbj2x" / "pbh2x")."""
    return next(float(r["time_vs_1x"]) for r in rows if r["preset"] == preset and int(r["trfc_ns"]) == MAIN_TRFC[preset]
                and int(r["nccdab"]) == n and r["refresh"] == mode)


def source_check():
    """AttAcc's own constants, when its sources are present (None otherwise)."""
    cfg = os.path.join(C.ATTACC, "src", "config.py")
    cpp = os.path.join(C.RAM, "src", "dram", "impl", "HBM3-PIM.cpp")
    if not (os.path.exists(cfg) and os.path.exists(cpp)):
        return None
    src = open(cfg).read()
    ba = re.search(r"ENERGY_TABLE\['PIM'\]\[PIMType\.BA\]\['mem'\]\s*=\s*\(([\d.]+)\s*\+\s*([\d.]+)\)\s*\*\s*8", src)
    alu = re.search(r"ENERGY_TABLE\['PIM'\]\[PIMType\.BA\]\['alu'\]\s*=\s*([\d.]+)", src)
    presets = {}
    for m in re.finditer(r'\{"(HBM3_[\d.]+Gbps(?:_NPC)?)",\s*\{([^}]*)\}', open(cpp).read()):
        presets[m.group(1)] = int(m.group(2).split(",")[14])      # nCCDAB column
    out = dict(act_pJ_bit=float(ba.group(1)), cell_pJ_bit=float(ba.group(2)), mac_pJ=float(alu.group(1)),
               nccdab={g: presets[v["preset"]] for g, v in GPUS.items()},
               nccdab_npc={g: presets[v["preset"] + "_NPC"] for g, v in GPUS.items()})
    out["ok"] = ((out["act_pJ_bit"], out["cell_pJ_bit"], out["mac_pJ"]) == (ACT_PJ_BIT, CELL_PJ_BIT, MAC_PJ)
                 and all(out["nccdab"][g] == v["n_attacc"] for g, v in GPUS.items())
                 and all(out["nccdab_npc"][g] == N_TIMING[v["preset"]] for g, v in GPUS.items()))
    return out


def table():
    with open(src("refresh_sweep.csv")) as f:
        rf = list(csv.DictReader(f))
    gpus = {}
    for g, v in GPUS.items():
        pr, n0, st = v["preset"], v["n_attacc"], v["stacks"]
        n_t = N_TIMING[pr]
        bw = bw_table(pr)
        B = int((v["capacity"] - WEIGHTS) // (SEQLEN * KV_TOKEN))
        even, busy = pairs(B, st)
        trfc = MAIN_TRFC[pr]
        gpus[g] = dict(
            preset=pr, data_rate_Gbps=RATE[pr] / 1000, tCK_ns=4000 / RATE[pr], stacks=st,
            mac_interval_clk=n0, mac_interval_ns=n0 * 4000 / RATE[pr], timing_limit_clk=n_t,
            timing_limit_ns=n_t * 4000 / RATE[pr],
            pim_bw_TBs_stack=bw[n0] / 1e12, pim_W_stack=E_BA * 1e-12 * bw[n0],
            pim_bw_timing_TBs_stack=bw[n_t] / 1e12, pim_W_timing_stack=E_BA * 1e-12 * bw[n_t],
            pim_W_stack_step_avg=E_BA * 1e-12 * bw[n0] * even / busy, pim_W_timing_stack_step_avg=E_BA * 1e-12 * bw[n_t] * even / busy,
            batch=B, pairs_mean=even, pairs_most_loaded=busy, kappa=busy / even,
            trfc_ns=trfc, r_gpu=(1 - trfc * 1e-9 / T_REFI) / (1 - 2 * trfc * 1e-9 / T_REFI),
            r_pim={str(n): r_pim(pr, n, rf) for n in sorted({n0, n_t})},
            r_pim_per_bank={m: {str(n): r_pim(pr, n, rf, m + "2x") for n in sorted({n0, n_t})} for m in ("pbj", "pbh")})

    # AttAcc's main.py, H100, one decode step at context 2048, per layer and per GPU
    with open(src("attacc_e2e_h100_llama65b.csv")) as f:
        e2e = {r["run"]: {k: float(x) for k, x in r.items() if k.endswith(("_ms", "_J"))}
               for r in csv.DictReader(f) if int(r["Lin"]) + int(r["Lout"]) - 1 == SEQLEN}
    h = gpus["H100"]
    kv = h["batch"] * SEQLEN * KV_TOKEN
    a_g, a_p = e2e["gpu_only"], e2e["pim_alternating"]
    attn_model = kv / (h["stacks"] * h["pim_bw_TBs_stack"] * 1e12 / h["kappa"]) / LAYERS * 1e6
    attacc = dict(
        fc_us=a_g["fc_ms"] / LAYERS * 1e3, attn_gpu_us=(a_g["attn_ms"] + a_g["softmax_ms"]) / LAYERS * 1e3,
        attn_pim_us=a_p["attn_ms"] / LAYERS * 1e3, attn_pim_model_us=attn_model,
        attn_pim_err_pct=100 * abs(attn_model / (a_p["attn_ms"] / LAYERS * 1e3) - 1),
        step_gpu_ms=a_g["step_ms"], step_pim_ms=a_p["step_ms"], speedup=a_g["step_ms"] / a_p["step_ms"],
        power_W_per_gpu=a_g["energy_J"] / TP / (a_g["step_ms"] / 1e3),
        dram_W_per_gpu=a_g["dram_energy_J"] / TP / (a_g["step_ms"] / 1e3))
    return dict(energy=dict(act_pJ_bit=ACT_PJ_BIT, cell_read_pJ_bit=CELL_PJ_BIT, in_bank_pJ_bit=ACT_PJ_BIT + CELL_PJ_BIT,
                            mac_pJ=MAC_PJ, mac_pJ_B=MAC_PJ / 2, pim_pJ_B=E_BA, pim_pJ_bit=E_BA / 8),
                model=dict(name="LLaMA-65B", tp=TP, context=SEQLEN, weights_B_per_gpu=WEIGHTS, kv_B_per_token=KV_TOKEN),
                gpus=gpus, attacc_e2e_h100=attacc)


def main():
    t = table()
    sc = source_check()     # printed only, so the committed JSON does not depend on whether AttAcc is checked out
    if sc is not None and not sc["ok"]:
        raise SystemExit(f"AttAcc's constants differ from the ones used here: {sc}")
    with open(C.out("pim_table.json"), "w") as f:
        json.dump(t, f, indent=1)
    print("GPU   MAC interval      PIM TB/s/stack  W/stack   tCCD_L TB/s  W     batch  kappa  r_gpu  r_pim")
    for g, v in t["gpus"].items():
        print(f"{g:5s} {v['mac_interval_clk']} clk {v['mac_interval_ns']:.2f} ns   {v['pim_bw_TBs_stack']:.2f}           "
              f"{v['pim_W_stack']:.1f}      {v['pim_bw_timing_TBs_stack']:.2f}        {v['pim_W_timing_stack']:.1f}  "
              f"{v['batch']:4d}   {v['kappa']:.3f}  {v['r_gpu']:.3f}  "
              + "/".join(f"{x:.3f}" for x in v["r_pim"].values()))
    a = t["attacc_e2e_h100"]
    print(f"PIM energy {t['energy']['pim_pJ_B']:.2f} pJ/B; AttAcc H100 PIM attention {a['attn_pim_us']:.1f} us/layer, "
          f"model {a['attn_pim_model_us']:.1f} us ({a['attn_pim_err_pct']:.2f} %); AttAcc sources: "
          f"{'not fetched, not compared' if sc is None else 'energy table and nCCDAB presets match'}")


if __name__ == "__main__":
    main()
