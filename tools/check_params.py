"""Keep the parameter tables and the README honest: every bound value printed in a table must equal the value in the
code, config or data file that the model actually uses.

Each binding names a table row (by ID, or by the start of its first cell), a text token that must appear in that row,
and the place the value lives. Every number in the token is compared with the value at the matching key after rounding
to the token's printed digits. A table edit without a code change, or a code change without a table edit, fails.

python3 tools/check_params.py        # exit 1 on any mismatch
"""
import ast
import csv
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NUM = re.compile(r"\d+(?:,\d{3})*(?:\.\d+)?(?:e-?\d+)?")

DP, TP, RD, PM = "dram/PARAMETERS.md", "thermal/PARAMETERS.md", "README.md", "PARAMETERS.md"
PT, I7, HS = "dram/results/pim_table.json", "dram/results/idd7_chips.json", "dram/results/hbm3e_split.json"
CFG, LCF = "thermal/geometry/h100_5stack.config", "thermal/geometry/h100_5stack/stack.lcf"
OP, RP = "thermal/data/inputs/beta_operating_point.json", "thermal/data/inputs/ripple_phases.json"
G3 = ("H100", "H200", "B200")


def g3(field):
    return [("json", PT, f"gpus.{g}.{field}") for g in G3]


def rq(*ids):
    return [("ref", "quick", i) for i in ids]


def rf(*ids):
    return [("ref", "full", i) for i in ids]


# (table, row, token, [source, ...]); a source is (kind, file or mode, key[, scale])
B = [
    # dram/: organisation, timing, refresh
    (DP, "D01", "5 / 6 / 8", g3("stacks")),
    (DP, "D02", "batch 94 / 185 / 244", g3("batch")),
    (DP, "D03", "5.2 / 6.4 / 8.0", g3("data_rate_Gbps")),
    (DP, "D04", "0.769 / 0.625 / 0.500", g3("tCK_ns")),
    (DP, "D12", "4 / 4 / 5", g3("timing_limit_clk")),
    (DP, "D13", "6 / 7 / 9", g3("mac_interval_clk")),
    (DP, "D13", "(4.6 / 4.4 / 4.5 ns)", g3("mac_interval_ns")),
    (DP, "D15", "350 (H100", [("json", PT, "gpus.H100.trfc_ns")]),
    (DP, "D15", "/ 410 (H200", [("json", PT, "gpus.H200.trfc_ns")]),
    (DP, "D16", "3.9", [("py", "dram/attacc/pim_table.py", "T_REFI", 1e6)]),
    (DP, "D18", "200 (H100", [("py", "dram/attacc/refresh_sweep.py", "TRFCPB['HBM3_5.2Gbps']")]),
    (DP, "D18", "/ 240 (H200", [("py", "dram/attacc/refresh_sweep.py", "TRFCPB['HBM3_6.4Gbps']")]),
    (DP, "D19", "1.157 / 1.168 / 1.169", [("json", PT, "gpus.H100.r_pim_per_bank.pbj.6"),
                                          ("json", PT, "gpus.H200.r_pim_per_bank.pbj.7"),
                                          ("json", PT, "gpus.B200.r_pim_per_bank.pbj.9")]),
    (DP, "D24", "1.064 (H100)", [("json", PT, "gpus.H100.kappa")]),
    (DP, "D24", "1.038 (H200)", [("json", PT, "gpus.H200.kappa")]),
    (DP, "D24", "1.049 (B200)", [("json", PT, "gpus.B200.kappa")]),
    (DP, "D25", "REPEAT 8", [("py", "dram/attacc/refresh_sweep.py", "REPEAT")]),
    (DP, "D26", "266 rows", [("csvrows", "dram/results/refresh_sweep.csv", "")]),
    (DP, "D27", "r_gpu 1.109", [("json", PT, "gpus.H100.r_gpu")]),
    (DP, "D27", "/ 1.133", [("json", PT, "gpus.H200.r_gpu")]),
    (DP, "D30", "31.00 us", [("json", PT, "attacc_e2e_h100.attn_pim_us")]),
    (DP, "D30", "31.17 us (0.54 %)", [("json", PT, "attacc_e2e_h100.attn_pim_model_us"),
                                      ("json", PT, "attacc_e2e_h100.attn_pim_err_pct")]),
    # dram/: energy and power
    (DP, "E01", "0.11", [("py", "dram/attacc/pim_table.py", "ACT_PJ_BIT")]),
    (DP, "E02", "0.44", [("py", "dram/attacc/pim_table.py", "CELL_PJ_BIT")]),
    (DP, "E04", "0.32 per FP16 MAC = 0.16", [("py", "dram/attacc/pim_table.py", "MAC_PJ"), ("json", PT, "energy.mac_pJ_B")]),
    (DP, "E05", "4.56 (0.57 pJ/bit)", [("json", PT, "energy.pim_pJ_B"), ("json", PT, "energy.pim_pJ_bit")]),
    (DP, "E09", "5.38 / 5.72 / 5.45", g3("pim_bw_TBs_stack")),
    (DP, "E09", "7.40 / 8.83 / 8.90", g3("pim_bw_timing_TBs_stack")),
    (DP, "E10", "24.5 / 26.1 / 24.8", g3("pim_W_stack")),
    (DP, "E10", "33.8 (H100", [("json", PT, "gpus.H100.pim_W_timing_stack")]),
    (DP, "E10", "23.1 and 31.7", [("json", PT, "gpus.H100.pim_W_stack_step_avg"),
                                  ("json", PT, "gpus.H100.pim_W_timing_stack_step_avg")]),
    (DP, "E11", "median 1.237 (1.141-1.576)", [("json", I7, "dynamic_ratio.median"), ("json", I7, "dynamic_ratio.min"),
                                               ("json", I7, "dynamic_ratio.max")]),
    (DP, "E12", "26.9 / 29.2 / 37.2", [("json", I7, f"gpus.H100.idd7_W_stack.hbm2_{c}") for c in ("lowest", "median", "highest")]),
    (DP, "E12", "1.10 / 1.19 / 1.51x", [("json", I7, f"gpus.H100.idd7_over_attacc.hbm2_{c}")
                                        for c in ("lowest", "median", "highest")]),
    (DP, "E12", "H200 41.2", [("json", I7, "gpus.H200.idd7_W_stack.hbm2_highest")]),
    (DP, "E12", "B200 40.7", [("json", I7, "gpus.B200.idd7_W_stack.hbm2_highest")]),
    (DP, "E13", "19.23 / 27.03 / 29.25", [("json", I7, f"hbm3_model.cases.{c}.stack_W")
                                          for c in ("data 0", "random, config knobs", "random")]),
    (DP, "E15", "read 4.257 = floor 2.629 + DQ 0.430 + TSV 0.615 + bank group 0.583",
     [("json", HS, k) for k in ("read_pJ_bit", "read_terms_pJ_bit.floor", "read_terms_pJ_bit.dq", "read_terms_pJ_bit.tsv",
                                "read_terms_pJ_bit.bg_bus")]),
    (DP, "E15", "in-stack 4.470 (whole row) / 5.937", [("json", HS, "patterns.whole_row.in_stack"),
                                                       ("json", HS, "patterns.four_reads.in_stack")]),
    (DP, "E16", "0.165 (0.072-0.234)", [("json", HS, "base_die_share.prior"), ("json", HS, "base_die_share.range[0]"),
                                        ("json", HS, "base_die_share.range[1]")]),
    (DP, "E17", "HBM2 6.14 (1.35x", [("json", I7, "hbm2_pim_energy.pim_pJ_B"), ("json", I7, "hbm2_pim_energy.pim_over_table")]),
    (DP, "E17", "4.11 / 5.73 / 6.20 (0.90 / 1.26 / 1.36x)",
     [("json", I7, f"hbm3_model.cases.{c}.pim_pJ_B") for c in ("data 0", "random, config knobs", "random")]
     + [("json", I7, f"hbm3_model.cases.{c}.pim_over_table") for c in ("data 0", "random, config knobs", "random")]),
    (DP, "E19", "23.899 (0.7468", [("json", "dram/results/bank_power_stack.json", "stack_W"),
                                   ("json", "dram/results/bank_power_stack.json", "simulated_pcs[0].W")]),
    (DP, "E21", "β 1.09 (1.03-1.13)", rf("beta.summary.every_bank", "beta.summary.every_bank_range[0]",
                                         "beta.summary.every_bank_range[1]")),
    (DP, "M02", "full 30.87 / 33.30 / 30.15", [("json", "dram/data/measured_copy_power.json", f"gpus.{g}.full_copy") for g in G3]),
    (DP, "M02", "idle 7.28 / 7.16 / 4.33", [("json", "dram/data/measured_copy_power.json", f"gpus.{g}.idle") for g in G3]),
    (DP, "M02", "dynamic 23.58 / 26.14 / 25.82", [("json", I7, f"gpus.{g}.copy_dynamic_W_stack") for g in G3]),
    (DP, "M12", "327,680", [("json", PT, "model.kv_B_per_token")]),
    (DP, "M14", "5.80", [("json", HS, "h200_measured.active_pJ_bit")]),
    # thermal/: geometry and materials
    (TP, "G01", "28.53 x 28.53", [("flp", "core_and_mem_ctrl.flp", "C_0:0", 1e3), ("flp", "core_and_mem_ctrl.flp", "C_0:1", 1e3)]),
    (TP, "G02", "x 8.50", [("flp", "core_and_mem_ctrl.flp", "C_0:2", 1e3)]),
    (TP, "G02", "y 4.02", [("flp", "core_and_mem_ctrl.flp", "C_0:3", 1e3)]),
    (TP, "G03", "45.53 x 36.57", [("flp", "interposer.flp", "I_0:0", 1e3), ("flp", "interposer.flp", "I_0:1", 1e3)]),
    (TP, "G04", "8.0 x 12.19", [("py", "thermal/geometry/build_5stack.py", "W_ST"), ("py", "thermal/geometry/build_5stack.py", "H_ST")]),
    (TP, "G07", "0.5 mm", [("py", "thermal/geometry/build_5stack.py", "GAP")]),
    (TP, "G09", "45.6 mm side, 1 mm, k 400", [("cfg", CFG, "s_spreader", 1e3), ("cfg", CFG, "t_spreader", 1e3), ("cfg", CFG, "k_spreader")]),
    (TP, "G10", "82.1 mm side, 6.9 mm, k 400, cp 3.55e6", [("cfg", CFG, "s_sink", 1e3), ("cfg", CFG, "t_sink", 1e3),
                                                          ("cfg", CFG, "k_sink"), ("cfg", CFG, "p_sink")]),
    (TP, "G11", "173.82", [("cfg", CFG, "c_convec")]),
    (TP, "G12", "0.02883", [("cfg", CFG, "r_convec")]),
    (TP, "G13", "t 0.6 mm, k 4.0, cp 4e6", [("cfg", CFG, "t_interface", 1e3), ("cfg", CFG, "k_interface"), ("cfg", CFG, "p_interface")]),
    (TP, "G15", "silicon, 50", [("lcf", LCF, "0:thickness", 1e6)]),
    (TP, "G16", "20 um, rho 0.25", [("lcf", LCF, "1:thickness", 1e6), ("lcf", LCF, "1:rho")]),
    (TP, "G17", "silicon, 50", [("lcf", LCF, "2:thickness", 1e6)]),
    (TP, "G18", "rho 0.007 m-K/W", [("lcf", LCF, "4:rho")]),
    (TP, "G18", "cp 1.75e6", [("lcf", LCF, "4:cp")]),
    (TP, "G19", "8 dies x 50 um, bonds 20 um", [("lcfcount", LCF, "mem_bank"), ("lcf", LCF, "4:thickness", 1e6),
                                                ("lcf", LCF, "3:thickness", 1e6)]),
    (TP, "G21", "20 um, rho 0.25", [("lcf", LCF, "19:thickness", 1e6), ("lcf", LCF, "19:rho")]),
    (TP, "G23", "rho 0.13", [("py", "thermal/geometry/build_5stack.py", "FILL", None, 1)]),
    (TP, "G23", "cp 2.875e6", [("py", "thermal/geometry/build_5stack.py", "FILL", None, 0)]),
    # thermal/: HotSpot settings, power and calibration
    (TP, "P05", "e = 3.5 %", [("py", "thermal/model/power.py", "E_TABLE", 100)]),
    (TP, "P07", "23.899 W per stack (bank.py constant 23.9022)",
     [("json", "thermal/data/inputs/dram_map_decode.json", "stack_W"), ("py", "thermal/steps/bank.py", "BASE_DRAM_W")]),
    (TP, "P09", "GPU 80 W, DRAM 23.9 W/stack, PIM 40 W/stack; dM 20",
     [("py", "thermal/steps/bank.py", k) for k in ("BASE_GPU_W", "BASE_DRAM_W", "PIM_W")] + [("json", OP, "dM")]),
    (TP, "P10", "no power constraint that beta gives: 1.872", rq("beta.summary.threshold_H100_e_x")),
    (TP, "P10", ", 3.04 at the matched", rq("recal.summary.best.e_x")),
    (TP, "P10", "starts at 1.84 pJ/bit", [("json", OP, "e_pJ_bit")]),
    (TP, "P10", "(P_G 332.7 W, P_M 80.1 W, P_P 52.1 W", [("json", OP, k) for k in ("P_G", "P_M", "P_P")]),
    (TP, "P10", "(1.836 / 1.529 / 1.899 pJ/bit)", rq(*(f"beta.summary.thresholds_e_x_at_1p11.{g}" for g in G3))),
    (TP, "P12", "80 / 480 W", [("py", "thermal/steps/calibrate.py", "G_LO"), ("py", "thermal/steps/calibrate.py", "G_HI")]),
    (TP, "P12", "memory share 0.1175", rq("measured.anchor.mem_per_board_W")),
    (TP, "P13", "half 1.51-1.80, quarter 1.75-2.86", rq("beta.summary.half[0]", "beta.summary.half[1]",
                                                       "beta.summary.quarter[0]", "beta.summary.quarter[1]")),
    (TP, "P13", "1.50-1.85 / 1.75-3.04", rf("bank.summary.half[0]", "bank.summary.half[1]",
                                           "bank.summary.quarter[0]", "bank.summary.quarter[1]")),
    (TP, "P14", "step 17.97 ms; PIM on 1.70 ms at 133.9 W/stack; off: GPU 359.1 W, memory 15.35",
     [("json", RP, k) for k in ("step_ms", "pim_on_ms", "pim_W_per_stack_on", "off[0]", "off[1]")]),
    (TP, "P14", "beta point (1.87) and below the matched one (3.04)", rq("beta.summary.threshold_H100_e_x",
                                                                          "recal.summary.best.e_x")),
    (TP, "S04", "64 x 64", [("py", "thermal/steps/recal.py", "GRID"), ("py", "thermal/steps/recal.py", "GRID")]),
    (TP, "S04", "1.082)", rf("beta.summary.every_bank_grid128")),
    (TP, "S06", "50", [("py", "thermal/model/geom.py", "AMB")]),
    (TP, "S06", "50", [("cfg", CFG, "ambient", 1, -273.15)]),
    (TP, "S08", "C_FACTOR 0.333", [("cdef", "thermal/hotspot/temperature.h", "C_FACTOR")]),
    (TP, "S14", "r grid 0.008-0.020", [("py", "thermal/steps/recal.py", "RS[0]"), ("py", "thermal/steps/recal.py", "RS[-1]")]),
    (TP, "S15", "k 0.75-2.0 (quick)", [("py", "thermal/steps/recal.py", "KS_QUICK[0]"), ("py", "thermal/steps/recal.py", "KS_QUICK[-1]")]),
    # thermal/: measured and fitted values
    (TP, "M01", "0.02259", rq("measured.h100_fit.s_host")),
    (TP, "M02", "0.18774 [0.168, 0.206]", rq("measured.h100_fit.s_rail", "measured.h100_fit.s_rail_ci[0]",
                                             "measured.h100_fit.s_rail_ci[1]")),
    (TP, "M05", "0.04328 K per board W; 0.1175", rq("measured.anchor.hbm_per_board_W", "measured.anchor.mem_per_board_W")),
    (TP, "M06", "0.701 [0.633, 0.779]", rq("measured.rowmiss.ratio", "measured.rowmiss.ratio_ci95[0]",
                                           "measured.rowmiss.ratio_ci95[1]")),
    (TP, "M07", "0.01361", rq("calibrate.k.k1.r_convec")),
    (TP, "M09", "beta 1.088, row-miss ratio 1.18", rq("recal.summary.x0.beta", "recal.summary.x0.ratio")),
    (TP, "M10", "beta 0.668, ratio 0.693", rq("recal.summary.x1.beta", "recal.summary.x1.ratio")),
    (TP, "M11", "x 0.975, k 1.927, r 0.01031, beta 0.674", rq("recal.summary.best.x", "recal.summary.best.k",
                                                              "recal.summary.best.r", "recal.summary.best.beta")),
    (TP, "M12", "x 0.74", rq("recal.summary.allowed.x[0]")),
    (TP, "M12", "beta 0.67-0.74", rq("recal.summary.allowed.beta[0]", "recal.summary.allowed.beta[1]")),
    (TP, "M13", "x 0.357", rq("recal.summary.basephy_target.x")),
    (TP, "M13", "beta 0.669", rq("recal.summary.basephy_target.beta")),
    (TP, "M14", "x 0.072 / 0.165 / 0.234", [("json", HS, "base_die_share.range[0]"), ("json", HS, "base_die_share.prior"),
                                            ("json", HS, "base_die_share.range[1]")]),
    (TP, "M14", "prior beta 0.94-1.04", rq("recal.summary.prior_range.beta[0]", "recal.summary.prior_range.beta[1]")),
    (TP, "M15", "1.1146 (range 1.0251-1.1513)", rf("bank.summary.f_bank", "bank.summary.f_bank_range[0]",
                                                   "bank.summary.f_bank_range[1]")),
    (TP, "M15", "quick mode 1.1205 (1.0234-1.1548)", rq("bank.summary.f_bank", "bank.summary.f_bank_range[0]",
                                                        "bank.summary.f_bank_range[1]")),
    (TP, "M15", "is 1.088 (1.026-1.132)", rq("beta.summary.every_bank", "beta.summary.every_bank_range[0]",
                                                  "beta.summary.every_bank_range[1]")),
    (TP, "S17", "it gives; 1.088 at", rq("beta.summary.every_bank")),
    (TP, "S17", "(f_bank 1.115 at", rf("bank.summary.f_bank")),
    (TP, "S17", "local slope 1.128 and", rq("recal.summary.x0.beta_local")),
    (TP, "M16", "1.74 (k", rq("bank.summary.base_die_over_dram")),
    (TP, "M16", "1.63 (best fit)", rq("recal.summary.best.base_over_dram")),
    (TP, "M17", "max 0.055 K", rq("bank.ripple.summary.per_layer_max_above_avg")),
    (TP, "M17", "burst bound 0.14", rq("bank.ripple.summary.per_layer_plus_burst_max")),
    (TP, "M17", "lumped 3.7", rq("bank.ripple.summary.lump_max_above_avg")),
    (TP, "M19", "0.93 K", rq("calibrate.k.k1.pim_peak_per_w_per_stack")),
    (TP, "S11", "matched beta 0.748 / 0.807", rq("recal.summary.die_average.bottom.beta",
                                                 "recal.summary.die_average.base.beta")),
    # PARAMETERS.md: assumptions that move beta (row = start of the first cell)
    (PM, "memory power split", "1.088 / 0.668", rq("recal.summary.x0.beta", "recal.summary.x1.beta")),
    (PM, "HBM temperature = hottest", "0.748 / 0.807", rq("recal.summary.die_average.bottom.beta",
                                                          "recal.summary.die_average.base.beta")),
    (PM, "base-die heat over", "0.669", rq("recal.summary.basephy_target.beta")),
    (PM, "slopes from one H100", "0.736", [("json", "thermal/checks/validation/e2_card2.json", "x1.beta_every_bank")]),
    (PM, "operating point at the H100", "0.674 against 0.643", rq("recal.summary.best.beta", "recal.summary.best.beta_fixed_e")),
    (PM, "PIM heat in every bank", "β 1.03-1.13", rf("beta.summary.every_bank_range[0]", "beta.summary.every_bank_range[1]")),
    (PM, "DRAM background power", "23.899 W/stack", [("json", "dram/results/bank_power_stack.json", "stack_W")]),
    # README: reproduced values (row = start of the first cell)
    (RD, "PIM bandwidth at the power-constrained", "5.38 / 5.72 / 5.45", g3("pim_bw_TBs_stack")),
    (RD, "PIM power there", "24.5 / 26.1 / 24.8", g3("pim_W_stack")),
    (RD, "PIM bandwidth at tCCD_L", "7.40 / 8.83 / 8.90", g3("pim_bw_timing_TBs_stack")),
    (RD, "PIM energy", "0.55 pJ/bit, 0.32 pJ/MAC", [("json", PT, "energy.in_bank_pJ_bit"), ("json", PT, "energy.mac_pJ")]),
    (RD, "busiest channel", "1.06", [("json", PT, "gpus.H100.kappa")]),
    (RD, "slowdown at doubled all-bank refresh", "1.109 / 1.133 / 1.133", g3("r_gpu")),
    (RD, "slowdown at doubled all-bank refresh", "1.109 / 1.135 / 1.133", [("json", PT, "gpus.H100.r_pim.6"),
                                                                  ("json", PT, "gpus.H200.r_pim.7"),
                                                                  ("json", PT, "gpus.B200.r_pim.9")]),
    (RD, "IDD7 stack power", "1.10-1.51x", [("json", I7, "gpus.H100.idd7_over_attacc.hbm2_lowest"),
                                            ("json", I7, "gpus.H100.idd7_over_attacc.hbm2_highest")]),
    (RD, "IDD7 stack power", "37.2 / 41.2 / 40.7", [("json", I7, f"gpus.{g}.idd7_W_stack.hbm2_highest") for g in G3]),
    (RD, "DRAM background power", "23.9 W", [("json", "dram/results/bank_power_stack.json", "stack_W")]),
    (RD, "HBM rise per", "0.0226 / 0.1877", rq("measured.h100_fit.s_host", "measured.h100_fit.s_rail")),
    (RD, "row-miss heating ratio", "0.70 (0.63-0.78)", rq("measured.rowmiss.ratio", "measured.rowmiss.ratio_ci95[0]",
                                                          "measured.rowmiss.ratio_ci95[1]")),
    (RD, "heat-sink resistance", "0.01361", rq("calibrate.k.k1.r_convec")),
    (RD, "temperature ripple", "0.055 K", rq("bank.ripple.summary.per_layer_max_above_avg")),
    (RD, "β, PIM at every bank, x = 0", "1.088 (1.026-1.132)", rq("beta.summary.every_bank", "beta.summary.every_bank_range[0]",
                                                                 "beta.summary.every_bank_range[1]")),
    (RD, "β, PIM in half", "1.51-1.80 / 1.75-2.86", rq("beta.summary.half[0]", "beta.summary.half[1]",
                                                       "beta.summary.quarter[0]", "beta.summary.quarter[1]")),
    (RD, "H100 threshold where β", "1.872", rq("beta.summary.threshold_H100_e_x")),
    (RD, "β at x = 1", "0.668", rq("recal.summary.x1.beta")),
    (RD, "comparison normalisations", "1.120 / 1.128", rq("bank.summary.f_bank", "recal.summary.x0.beta_local")),
    (RD, "comparison normalisations", "- / 0.661", rq("recal.summary.best.beta_local")),
    (RD, "x matching the row-miss ratio", "0.975; 0.674 (0.668-0.736)",
     rq("recal.summary.best.x", "recal.summary.best.beta", "recal.summary.allowed.beta[0]", "recal.summary.allowed.beta[1]")),
    (RD, "same, die-average readout", "0.686 / 0.517; 0.748 / 0.807",
     rq("recal.summary.die_average.bottom.x", "recal.summary.die_average.base.x",
        "recal.summary.die_average.bottom.beta", "recal.summary.die_average.base.beta")),
    (RD, "same, base-die heat on the GPU-side half", "0.357; 0.669",
     rq("recal.summary.basephy_target.x", "recal.summary.basephy_target.beta")),
]


def rows(path):
    out = {}
    for ln in open(os.path.join(ROOT, path), encoding="utf-8"):
        if ln.startswith("|") and not set(ln.strip()) <= set("|-: "):
            cells = [c.strip() for c in re.split(r"(?<!\\)\|", ln.strip().strip("|"))]
            out.setdefault(cells[0], []).append(ln)
    return out


def dig(obj, key):
    if isinstance(key, tuple):                                    # keys that contain dots
        for part in key:
            obj = obj[part]
        return obj
    for part in re.findall(r"[^.\[\]]+|\[-?\d+\]", key):
        obj = obj[int(part[1:-1])] if part.startswith("[") else obj[part]
    return obj


_PY = {}


def pyval(path, expr):
    if path not in _PY:
        env = {}
        for node in ast.parse(open(os.path.join(ROOT, path), encoding="utf-8").read()).body:
            if isinstance(node, ast.Assign):
                try:
                    v = ast.literal_eval(node.value)
                except ValueError:
                    try:
                        v = eval(compile(ast.Expression(node.value), path, "eval"), {"__builtins__": {}}, dict(env))
                    except Exception:
                        continue
                for t in node.targets:
                    if isinstance(t, ast.Name):
                        env[t.id] = v
                    elif isinstance(t, ast.Tuple) and isinstance(v, tuple) and len(t.elts) == len(v):
                        env.update({e.id: x for e, x in zip(t.elts, v) if isinstance(e, ast.Name)})
        _PY[path] = env
    name, rest = re.match(r"(\w+)(.*)", expr).groups()
    v = _PY[path][name]
    for part in re.findall(r"\[([^\]]+)\]", rest):
        v = v[ast.literal_eval(part)]
    return v


def lcf_layers(path):
    lines = [ln.strip() for ln in open(os.path.join(ROOT, path)) if ln.strip() and not ln.startswith("#")]
    return [dict(zip(("layer", "lateral", "power", "cp", "rho", "thickness", "flp"), lines[i:i + 7]))
            for i in range(0, len(lines), 7)]


def value(src):
    kind, f, key = src[:3]
    scale = src[3] if len(src) > 3 and src[3] is not None else 1
    if kind == "json":
        v = dig(json.load(open(os.path.join(ROOT, f))), key)
    elif kind == "ref":
        refs = {e["id"]: e for e in json.load(open(os.path.join(ROOT, "thermal/results/reference.json")))["values"]}
        v = refs[key][f]
    elif kind == "py":
        v = pyval(f, key)
        if len(src) > 4:                                      # field of a tab-separated string constant
            v = float(v.split("\t")[src[4]])
    elif kind == "cfg":
        m = re.search(rf"^\s*-{key}\s+(\S+)", open(os.path.join(ROOT, f)).read(), re.M)
        v = float(m.group(1)) + (src[4] if len(src) > 4 else 0)
    elif kind == "lcf":
        i, field = key.split(":")
        v = float(lcf_layers(f)[int(i)][field])
    elif kind == "lcfcount":
        v = sum(key in L["flp"] for L in lcf_layers(f))
    elif kind == "flp":
        unit, i = key.split(":")
        for ln in open(os.path.join(ROOT, "thermal/geometry/h100_5stack", f)):
            p = ln.split()
            if p and p[0] == unit:
                v = float(p[1 + int(i)])
    elif kind == "csvrows":
        v = sum(1 for _ in csv.DictReader(open(os.path.join(ROOT, f))))
    elif kind == "cdef":
        v = float(re.search(rf"#define\s+{key}\s+(\S+)", open(os.path.join(ROOT, f)).read()).group(1))
    return float(v) * scale


def agrees(token_num, actual):
    t = token_num.replace(",", "")
    if "e" in t:
        return abs(actual - float(t)) <= 1e-9 * abs(float(t))
    nd = len(t.split(".")[1]) if "." in t else 0
    return abs(actual - float(t)) <= 0.5 * 10 ** -nd + 1e-12


def main():
    tables, fails = {}, 0
    for table, row, token, srcs in B:
        if table not in tables:
            tables[table] = rows(table)
        lines = [ln for c, lns in tables[table].items() if c == row or (table not in (DP, TP) and c.startswith(row)) for ln in lns]
        where = f"{table} {row!r}"
        hit = [ln for ln in lines if token in ln]
        if not hit:
            print(f"FAIL {where}: token {token!r} not in the row" + ("" if lines else " (row missing)"))
            fails += 1
            continue
        nums = NUM.findall(re.sub(r"\b[A-Za-z_]+\d+\b", " ", token))   # H100, FP16 are names, not values
        if len(nums) != len(srcs):
            print(f"FAIL {where}: token {token!r} has {len(nums)} numbers for {len(srcs)} sources")
            fails += 1
            continue
        for n, s in zip(nums, srcs):
            try:
                a = value(s)
            except Exception as e:                                # missing key or file
                print(f"FAIL {where}: {s[1]} {s[2]}: {e!r}")
                fails += 1
                continue
            if not agrees(n, a):
                print(f"FAIL {where}: table {n} vs {s[1]} {s[2]} = {a:.6g}")
                fails += 1
    n = sum(len(s) for *_, s in B)
    print(f"check_params: {n - fails}/{n} values agree ({len(B)} table entries)")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
