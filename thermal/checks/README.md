# checks: evidence behind the sensitivities in PARAMETERS.md

Results of the HotSpot audit of 2026-11-11, with the scripts that made them. The scripts are records: they ran in the
audit folder (`<audit>`) against the analysis repository's HotSpot runner (`<analysis>/pipeline/e2e_joint`:
`hotspot_recal.py`, `thermal_pim_sim.py`, `thermal_bank_pim.py`; same model code as `thermal/model`) and the same HotSpot
source as `thermal/hotspot`. Their helpers are included (`numerics/common.py`, `numerics/analyze.py`,
`numerics/energy.py`, `variants/_common.py`); the analysis-repository modules they import are not, so they do not run
from this repository. Inputs of each script are named in its docstring or first lines. Two run here:
`validation/meas_t63.py` (`python3 -m thermal.checks.validation.meas_t63`, data in `data/measured/`) and
`numerics/tol_summary.py` (needs the audit runs).

IDs refer to `thermal/PARAMETERS.md`. "x = 0" is the joint fit with all memory power in the DRAM dies (k 0.954,
r 0.01102), "best" the fit matched to the row-miss ratio (x 0.975, k 1.927, r 0.01031). Operating point of every
beta below: `data/inputs/beta_operating_point.json`. The beta values in these records use the local-slope
normalisation (PIM rise per W over the local memory slope at that fixed point); they are kept as the evidence for the
relative effects quoted in PARAMETERS.md. The model's beta (one definition, `model/beta.py`) is in `out/beta.json`.

| file | shows | made by | IDs |
|---|---|---|---|
| `numerics/out_e_readout_fd_g64.json` | beta at x = 0 and best under the local, regression and amplitude normalisations, per readout; finite-difference width dM 0.5-40 W | `numerics/step_e.py` | P09, S17 |
| `numerics/out_superpos.json` | superposed unit responses against one joint run at the operating point: max 0.009 K | `numerics/superpos.py` | P11, S07 |
| `numerics/tol_summary.json` | iterative solver stopping at 1e-6 / 1e-8 against 1e-10 K: max 0.01 K (one output step) | `numerics/tol_summary.py` on runs of `numerics/run2.py` | S03 |
| `numerics/eb_bank.json` | energy balance of the bank-model runs, +2.3 to +2.8 % | `numerics/eb_bank.py`, `numerics/energy.py` | S05 |
| `numerics/out_sym_refit.json` | refit with symmetric (harmonic-mean) lateral conductance: beta -0.4 % at x = 0, 0.12-0.25 % at the fit | `numerics/sym_refit.py` | S05 |
| `numerics/out_conv_x0.json` | grid 32 / 64 / 128 at x = 0: beta 1.1349 / 1.1282 / 1.1206 | `numerics/conv_pts.py` | S04 |
| `numerics/out_best_xscan_g64_g128.json` | x scan at the best k on grid 64 and 128: matched x 0.972 / 0.901, beta 0.661 / 0.663 | inline computation in the audit, no script kept | P04, S04 |
| `numerics/out_peak2_g64.json`, `out_peak2_g128.json` | location of the hottest cell (0.14-0.36 mm from the die edge) and die-average readouts | `numerics/peak2.py` | S11 |
| `numerics/out_step1_direct_vs_interp.json` | direct HotSpot runs at the fitted (k, r) against the quadratic-in-r interpolation: <= 0.3 % | `numerics/step1.py` | S14 |
| `validation/meas_t63.json` | measured 63 % times after a GEMM power step: HBM 14.2-18.2 s, GPU 6.1-8.1 s | `validation/meas_t63.py` (data in `data/measured/h100_exp1/h100_transient_dt2_r*.csv`) | S08 |
| `validation/trans_cmp_match.json` | model step response at the best fit: HBM 63 % at 0.46 s | `validation/trans_cmp.py` | S08 |
| `validation/e3_ripple_c.json` | ripple with HotSpot's capacitance and with 3x (physical) capacitance: 0.055 / 0.018 K | `validation/e3_ripple_c.py` | M17, S08 |
| `validation/e2_card2.json` | the recalibration repeated with the slopes of a second H100 card: beta 0.736 at x = 1 (the measured ratio needs x 1.21, outside 0-1) | `validation/e2_card2.py` | M12 |
| `spreader/spreader_result.json` | spreader 45.6 against 70 mm: bank / uniform ratio -1.1 %, peak per W +0.7 % | `spreader/setup.py`, `spreader/analyze.py` | G09 |
| `geometry/summary_result.json` | package and layer variants (interposer, base die, DRAM-die and bond thickness, TIM, filler, bottom bond, secondary path, vapor chamber): change of beta at x = 0 and at the fit | `geometry/summary.py` | G07, G15-G19, G21-G25 |
| `variants/package_variants.json` | the base model and four variants refitted (GPU power in the central 70 %: matched beta 0.671; thicker TIM over the GPU: 0.652; bottom path on: 0.663; HotSpot-HMS-like layers: 0.704); beta over ratio 0.93-0.96 | `variants/package_variants.py`, `variants/_common.py` (ran in `<analysis>/pipeline/e2e_joint/hotspot_checks/`) | G01, P01, P03, P06 |
| `variants/gpu_die.json` | GPU-die slope of the model against the measurement (-13 to -15 %) along the fit family | `variants/gpu_die.py` (same origin) | M04, P01 |
| `geometry/bank128_result.json` | 64 two-bank cells against 128 one-bank cells per die: f_bank 1.118 -> 1.090 / 1.107 | `geometry/bank128.py` | F01 |
| `skeptic/op_result.json` | the operating point rerun from the analysis model and beta at 1.84 / 2.5 / 3.1 pJ/bit | `skeptic/check_op.py` | P10 |
| `rconvec/summary.json` (see `rconvec/README.md`) | r_convec 0.011-0.2 against the measured slopes and temperatures; larger spreader/sink refits: r 0.004-0.006, beta 1.085-1.087 | `rconvec/` scripts | M07, G09, G10 |
