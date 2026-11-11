# Data sources

## measured/ (H100 SXM 80GB, from a private measurement repository)

| file | source | sha256 |
|---|---|---|
| `h100_exp1/h100_anchor_r1.csv` | commit `c7f4e29`, `experiments/exp1_coupling/`, same name, unchanged | `fa44975f…e70a` |
| `h100_exp1/h100_anchor_r2.csv` | same | `34904a93…abcd` |
| `h100_exp1/h100_anchor_r3desc.csv` | same | `f215f81a…197a` |
| `h100_exp1/h100_combo2d.csv` | same | `c97bc63a…8f81` |
| `h100_exp1/h100_combo2d_g0p3.csv` | same | `af89e03e…0968` |
| `h100_exp1/h100_memsweep.csv` | same | `a34b6c71…150e` |
| `h100_exp1/h100_memsweep_dense.csv` | same | `88603afc…7695` |
| `h100_exp1/h100_transient_dt2_r1.csv` | same (GEMM power step, 2 s sampling; used only by `checks/validation/meas_t63.py`) | `421e2289…3e22` |
| `h100_exp1/h100_transient_dt2_r2.csv` | same | `734d635c…0b6d` |
| `h100_exp1/h100_transient_dt2_r3.csv` | same | `cd1af67a…05a0` |
| `h100_rowmiss_rows.csv` | commit `38e0926`, `261110_rowmiss_kit/out/<run>/rowmiss/01_rows.csv` (source sha256 `b4fb0fe2…efaa8`); 15 of its columns kept, no host or GPU identifiers | `5c21cb66…d024` |

Full hashes: `shasum -a 256 thermal/data/measured/*.csv thermal/data/measured/h100_exp1/*.csv`.

- `h100_exp1`: one row per settled power step of a GEMM or memory-copy duty sweep. `power` board W (nvidia-smi
  power.draw), `mem_power_w` memory W ("GPU Memory Power" of `nvidia-smi -q -d POWER`, NVIDIA's 1 s average of the
  memory subsystem, median of 3 reads), `thbm` / `tgpu` HBM and GPU temperature (degC, nvidia-smi), `settled` = 1 if the
  step reached steady state. Non-memory power P_G = power - mem_power_w.
  Used: 89 settled non-idle rows (HBM fit), 60 settled GEMM rows of the three anchor files (calibration target).
- `h100_rowmiss_rows.csv`: streaming reads (`pattern` seq) and reads that open a new row at every access (`rowmiss`)
  at matched read rates (`pair`), plus idle and FMA-only rows. `board_w` (from the GPU energy counter), `mem_power_w`, `thbm` are window means;
  `mem_gbps` the achieved read rate. Used: the 38 rows with `valid` and `settled` true.

## inputs/ (values from the DRAM model of `dram/` and the GPU-side decode model of the analysis repository)

| file | content |
|---|---|
| `dram_map_decode.json` | DRAM power per 2 x 2 block (W) of the calibration runs: DRAMPower per-bank energy of an H100 Llama-3.1-8B decode trace |
| `beta_operating_points.json` | GPU+PIM operating points over the in-bank read energy e = 0-6 pJ/bit (step 0.01) for H100, H200 and B200, no power constraint: P_G, P_M (idle included), P_P; GPU-only P_G, P_M; idle memory m0, heat share phi, measured slopes a, b. Used by the beta fixed point (`model/beta.py`). Analysis repository commit `cc8fdf6`, `pipeline/e2e_joint/abs_threshold.py`: `point(g, e, "timing")`, `GO[g]["split"]`, `IDLE[g][1]`, `CARD[g]["rail_heat"]`, `coef_model(g)`; exported by `tools/export_beta_ops.py` from a snapshot of that commit (measurement repositories pim_thermal 38e0926, pim_thermal_h100b 4f56365, pim_thermal_b200b bc2adb9, pim_thermal_h100x8 cd7d24a) |
| `beta_operating_point.json` | GPU, memory and PIM power (W) at the H100 threshold at beta 1.11 (1.84 pJ/bit): the fixed point of the local-slope beta (comparison only) and the start of the beta fixed-point iteration; computed in the analysis repository, inputs in `gpu_side_calibration.json` |
| `ripple_phases.json` | decode-step length, PIM-on time, layers and phase powers for the ripple study, with their derivation (s, ek, mu_K, rail_heat) |
| `gpu_side_calibration.json` | the GPU-side calibration values behind the two files above (decode energies, idle powers, rail heat share, PIM bandwidth), each with its analysis-repository file, key and commit; not recomputed here |
| `hbm3e_read_split.json` | HBM3E read energy split (pJ/bit) = `dram/results/hbm3e_split.json`, from HBM-Power main 61aada9 `config/HBM3E_6400MTs/power.json` with the v0.3-alpha timing and organisation files; our I/O-and-TSV split, used only for the reference x (not in the paper) |
