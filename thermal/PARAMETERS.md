# thermal/ parameters and their sources

Every value, setting and modelling choice of the thermal model, with the source it was checked against. Each entry was
checked twice against the primary source (paper section, page, table or figure; JESD238B.01 by section and table only;
HotSpot 6.0 source at 47b1e83 by file and line; our measurement files); derived values were recomputed and sensitivity
runs repeated where noted. Paths in backticks are relative to `thermal/`. `checks/` holds the audit runs behind the
sensitivities ([checks/README.md](checks/README.md)). "Analysis repository" marks a value computed outside this
repository (listed in `data/inputs/gpu_side_calibration.json` where the model uses it).

Status:
- verified: value, unit and source agree; derived values recompute exactly.
- verified, note: value confirmed; a caveat applies (note column).
- assumption: no direct source; the note gives the reason and, where run, the effect on beta.
- documented: a known deviation kept on purpose, stated where it is set.

"Analysis repository, not included" marks a value computed in our private analysis repository from our GPU
measurements; the note says which measurement. The values the model reads are copied into `data/inputs/`.

beta is the one definition of `model/beta.py` (README, "β"): the memory power that raises the peak DRAM temperature as
much as the PIM power does, over the PIM power, at the H100 GPU+PIM threshold point that this beta gives. "matched beta"
is beta at the base-die share x that reproduces the measured row-miss ratio (0.674 with the hottest-cell readout). Three
models appear below: the k = 1 model of steps 2-3 (calibration, bank model, ripple), the x = 0 fit (k 0.954, all memory
heat in the DRAM dies; the highest beta, shown in the heatmaps) and the matched fit (x 0.975, k 1.927). The variant runs
in `checks/` (geometry, package, numerics, second card) used the local-slope normalisation (comparison only, S17); their
effects are quoted as recorded and are relative changes of that normalisation.

## Assumptions and their effect on beta

| assumption | alternative tried | effect | source |
|---|---|---|---|
| bond conductivity k = 1 in steps 2-3 (calibration, bank model) | joint fits k 0.95-1.95 | x = 0 fit (k 0.954): beta 1.09; matched 0.674 (x in the CI: 0.668-0.736); one beta definition | `out/beta.json`, `out/recal.json` |
| HBM temperature = hottest DRAM grid cell | die-average sensor: bottom DRAM die / base die | matched beta 0.748 / 0.807 | `out/recal.json` die_average |
| memory power uniform over the DRAM dies, x over the whole base die | x over the GPU-side half of the base die | matched beta 0.669 | `out/recal.json` basephy_target |
| GPU power uniform over a square die | power in the central 70 % | matched beta 0.664 -> 0.671 | `checks/variants/package_variants.json` core70 |
| MAC share 3.5 % at the column decoders | 0.9 % / 10 % / 30 % | beta +0.3 % / -0.7 % / -3.0 % (x = 0) | `out/beta.json` floorplan_mac |
| bank arrays 50 % of the die, stripe parallel to the GPU edge | 75 %; perpendicular stripe | beta 1.03-1.13 over all cases (x = 0) | `out/beta.json` floorplan_mac |
| 64 two-bank cells per die | 128 one-bank cells | f_bank 1.118 -> 1.090-1.107 | `checks/geometry/bank128_result.json` |
| top cooling only | bottom path 0.5 K/W, 60 mm substrate | matched beta -0.001 | `checks/geometry/summary_result.json` sec |
| HotSpot-HMS layer thicknesses | interposer 100 µm, base die 100 µm, DRAM die 35 µm, top die 135 µm | matched beta -0.007 to +0.005 | `checks/geometry/summary_result.json` |
| top TIM 20 µm, rho 0.25 m-K/W | 56 µm everywhere; 14 mm²K/W over the HBM only | matched beta 0.655 / 0.676 (local slope; base 0.664) | `checks/geometry/summary_result.json` |
| spreader k 400, 1 mm | vapor chamber (k 4000, 3 mm) | matched beta +0.018 to +0.039 | `checks/geometry/summary_result.json` |
| spreader side 45.6 mm | 70 mm | f -1.1 % | `checks/spreader/spreader_result.json` |
| filler, off-stack bond, bottom bond | k 1, rho 1.0 | below 0.001 | `checks/geometry/summary_result.json` mold1, ubump1 |
| grid 64 for the recalibration | grid 128 | beta(x=0) -0.7 %, matched beta +0.3 %, matched x 0.975 -> 0.90 | `checks/numerics/out_conv_x0.json`, `out_best_xscan_g64_g128.json` |
| HotSpot's asymmetric lateral conductance (energy balance +2.3 to +2.8 %) | symmetric build | matched beta within 0.25 % | `checks/numerics/out_sym_refit.json` |
| iterative solver, tolerance 1e-6 K | 1e-8, 1e-10 | 0.01 K (one output step) | `checks/numerics/tol_summary.json` |
| quadratic interpolation in r | direct runs at the fitted (k, r) | <= 0.3 % | `checks/numerics/out_step1_direct_vs_interp.json` |
| finite-difference width dM 20 W | 0.5-40 W | within 0.4 % up to 20 W; +2.2 % at 40 W | `checks/numerics/out_e_readout_fd_g64.json` |
| operating point: the H100 threshold this beta gives (fixed point) | fixed 1.84 pJ/bit | matched beta 0.674 against 0.643 | `out/recal.json` summary.best |
| slopes from one H100 card | second H100 card | beta 0.736 at x = 1 (local slope; this card 0.654) | `checks/validation/e2_card2.json` |
| stack footprint 8.0 x 12.19 mm (HBM3E about 10 x 11 mm), dark site top right, zero gap within a stack column | no sensitivity run | | G04-G06 |

## Package geometry and materials

| ID | Parameter | Value | Unit | Source (location) | Status | Note |
|---|---|---|---|---|---|---|
| G01 | GPU die | 28.53 x 28.53 mm (814 mm2), power uniform | mm | [H100 whitepaper](https://resources.nvidia.com/en-us-tensor-core/gtc22-whitepaper-hopper) p.17 (814 mm²) | verified, note | square shape and uniform power are ours; central-70 % power: matched beta 0.664 -> 0.671 |
| G02 | GPU die position | x 8.50-37.03 mm, y 4.02-32.55 mm (centred on stack columns) | mm | `geometry/build_5stack.py` | assumption | our placement (`geometry/build_5stack.py`); no sensitivity run |
| G03 | Interposer | 45.53 x 36.57 mm, no overhang | mm | `geometry/build_5stack.py`; HotSpot-HMS model files ([Hong et al.](https://arxiv.org/abs/2403.09358)) | verified, note | Same convention as HotSpot-HMS (no overhang); bottom path off |
| G04 | HBM stack footprint | 8.0 x 12.19 mm (97.5 mm2) | mm | [Hong et al.](https://arxiv.org/abs/2403.09358) Table II (die 96 mm²); HotSpot-HMS model files ([Hong et al.](https://arxiv.org/abs/2403.09358)); [ServeTheHome](https://www.servethehome.com/sk-hynix-hbm-packaging-at-hot-chips-2026/) (HBM3E about 10 x 11 mm) | assumption | no sensitivity run; beta is a ratio of two responses in the same stack |
| G05 | Stack sites | six sites, 3 + 3 on the left and right die edges, zero gap within a column |  | [H100 whitepaper](https://resources.nvidia.com/en-us-tensor-core/gtc22-whitepaper-hopper) p.18 (6 sites, 12 controllers), p.19 Fig. 6 (controllers on two edges) | verified, note | Fig. 6 is a block diagram, not to scale; zero gap within a column untested |
| G06 | Powered stacks and dark site | 5 powered; dark site top right, unpowered silicon; dark bond tile k 4 |  | [H100 whitepaper](https://resources.nvidia.com/en-us-tensor-core/gtc22-whitepaper-hopper) p.18 (5 HBM3 stacks) | assumption | no sensitivity run for the dark-site position or dummy-as-silicon; they decide the hottest stack, not the per-stack beta |
| G07 | GPU-stack gap | 0.5 mm filled with filler (k 7.7) | mm | HotSpot-HMS model files ([Hong et al.](https://arxiv.org/abs/2403.09358)) (0.6 mm gap); `checks/geometry/summary_result.json` mold1 | assumption | filler k 1: beta(x=0) -0.0008, matched -0.0002 |
| G08 | Stack orientation | 8.0 mm toward the GPU, 12.19 mm along the GPU edge | mm | HotSpot-HMS model files ([Hong et al.](https://arxiv.org/abs/2403.09358)) | verified, note | HotSpot-HMS model files are not public; Hong et al. Table II gives only the materials |
| G09 | Heat spreader | 45.6 mm side, 1 mm, k 400 W/m-K, cp 3.55e6 |  | HotSpot-HMS model files ([Hong et al.](https://arxiv.org/abs/2403.09358)); [Hong et al.](https://arxiv.org/abs/2403.09358) Table II (spreader 400 W/m-K); HotSpot temperature_grid.c:801-808; `checks/spreader/spreader_result.json` | verified, note | 45.6 mm is the smallest side HotSpot accepts; 70 mm spreader: bank/uniform -1.1 %; vapor chamber: matched beta +0.018 to +0.039 |
| G10 | Heat sink | 82.1 mm side, 6.9 mm, k 400, cp 3.55e6 |  | HotSpot-HMS model files ([Hong et al.](https://arxiv.org/abs/2403.09358)) (thickness, material); side = 1.8 x spreader (our rule) | verified, note | enters steady results only through the fitted r_convec |
| G11 | Sink heat capacity c_convec | 173.82 J/K (rule for 82.1 mm sink: 262.88) | J/K | `geometry/h100_5stack.config` c_convec (comment) | documented | kept from the 1-stack setup; the rule 140.4 x (s/0.06)² gives 262.88; transient only, ripple bound conservative |
| G12 | Placeholder r_convec in the config | 0.02883 K/W (overwritten per run) | K/W | `model/geom.py` variant() overwrites it | verified, note | Overwritten in every run; never used in a result |
| G13 | Interface material entries | t 0.6 mm, k 4.0, cp 4e6 (unused with a layer file) |  | HotSpot temperature_grid.c:338-371, 649-655 | verified, note | unused with a layer file (stated in the config comment); the TIM is `stack.lcf` layer 19 |
| G14 | Chip entries t_chip, k_chip | HotSpot defaults (unused with a layer file) |  | [HotSpot](https://github.com/uvahotspot/HotSpot/tree/47b1e83c19) | verified |  |
| G15 | Interposer layer | silicon, 50 | um | HotSpot-HMS model files ([Hong et al.](https://arxiv.org/abs/2403.09358)) (100 µm); `checks/geometry/summary_result.json` interp100 | assumption | 100 µm: beta(x=0) -0.009, matched -0.007 to +0.005 |
| G16 | Bottom bond (dies to interposer) | 20 um, rho 0.25 m-K/W (k 4) |  | HotSpot temperature.c:46-48; [CoMeT create.py:12-13](https://github.com/marg-tools/CoMeT/blob/250ba95d1b3fde119766e92ef347226514c541e8/floorplanlib/create.py#L12-L13); `checks/geometry/summary_result.json` ubump1 | assumption | rho 1.0: beta < 0.001 |
| G17 | Base die | silicon, 50 | um | HotSpot-HMS model files ([Hong et al.](https://arxiv.org/abs/2403.09358)); `checks/geometry/summary_result.json` base100 | assumption | 100 µm: matched beta -0.004 |
| G18 | Silicon conductivity | rho 0.007 m-K/W (k 142.9 W/m-K), cp 1.75e6 |  | [Hong et al.](https://arxiv.org/abs/2403.09358) Table II (141 W/m-K); HotSpot temperature.c:26 | verified | 1/0.007 = 142.9, 1.3 % above 141 |
| G19 | DRAM die and bond thickness | 8 dies x 50 um, bonds 20 um; stack 630 | um | HotSpot-HMS model files ([Hong et al.](https://arxiv.org/abs/2403.09358)) (dies 50 µm, bonds 19 µm, top die 135 µm); `checks/geometry/summary_result.json` topdie, die35 | verified, note | thickness variants: beta(x=0) up to +0.011, matched up to +0.004 (the combined HotSpot-HMS-like variant also doubles the TIM: -0.010) |
| G20 | Bond conductivity k (fitted) | 0.95-1.95 W/m-K (best fit 1.93); thickness 20 um | W/m-K | [Xylem](https://iacoma.cs.uiuc.edu/iacoma-papers/micro17_2.pdf) Sec. 2.5 p.3; `out/recal.json` family | verified, note | fitted range brackets the literature 1.5 |
| G21 | Top TIM | 20 um, rho 0.25 m-K/W (5 mm2K/W) |  | HotSpot-HMS model files ([Hong et al.](https://arxiv.org/abs/2403.09358)); `checks/geometry/summary_result.json` tim2, tim14, timH14 | verified, note | no measurement of the TIM over the HBM; 56 µm everywhere: matched 0.655 (unclamped); 14 mm²K/W over HBM only: 0.676 (local slope; base 0.664) |
| G22 | Silicon column over the GPU | GPU footprint silicon in all die and bond layers 3-17 |  | HotSpot-HMS model files ([Hong et al.](https://arxiv.org/abs/2403.09358)) | verified, note | HotSpot-HMS model files are not public; Hong et al. Table II gives only the materials |
| G23 | Filler | rho 0.13 m-K/W (k 7.7), cp 2.875e6 |  | [CoMeT create.py:15-16](https://github.com/marg-tools/CoMeT/blob/250ba95d1b3fde119766e92ef347226514c541e8/floorplanlib/create.py#L15-L16); `checks/geometry/summary_result.json` mold1 | verified, note | Insensitive (< 0.001) |
| G24 | Off-stack bond tiles | rho 0.25 m-K/W |  | HotSpot-HMS model files ([Hong et al.](https://arxiv.org/abs/2403.09358)) (gap tile rho 0.25); HotSpot temperature.c:47 | assumption | Off-stack materials < 0.001 |
| G25 | Secondary heat path | off (top cooling only) |  | HotSpot temperature.c:450-451, 461-462; `checks/geometry/summary_result.json` sec | verified, note | off as in HotSpot-HMS; bottom path (0.5 K/W, 60 mm substrate): matched beta -0.001; the config comment cites temperature.c:450-451, 461-462 |
| F01 | Banks per DRAM die | 128 banks (two channels x 64) as 64 two-bank cells |  | JESD238B.01 Sec. 2, Table 4; [AttAcc](https://doi.org/10.1145/3620665.3640422) Sec. 4.1 | verified, note | code, README and die figure count 64 two-bank cells; a 128-cell floorplan gives f_bank 1.090-1.107 against 1.118 (f_bank normalisation, comparison only; `checks/geometry/bank128_result.json`) |
| F02 | Bank-array area fraction | 50 % (75 % alternative) | % | [HBM-PIM](https://doi.org/10.1109/ISCA52012.2021.00013) Sec. VI p.8, Tables IV-V | verified, note | half of the sub-arrays removed for PIM units: about 54 % (our arithmetic); 75 %: f 1.044 |
| F03 | Central TSV stripe | 1.6 mm wide (a50), 1.0 mm (a75) | mm | [FGDRAM](https://doi.org/10.1145/3123939.3124545) Sec. 2.1 p.3; HotSpot-HMS model files ([Hong et al.](https://arxiv.org/abs/2403.09358)) (TSV 1.72 mm) | assumption | Folded into the a50/a75 range |
| F04 | Bank cell layout | array 50.0 %, column decoder 13.2 %, row decoder 16.8 %, stripe 20.0 % (a50) | % | `model/geom.py` die_blocks() | assumption | our layout (`model/geom.py`); array share 75 % and the perpendicular stripe: beta 1.026-1.132 at x = 0 over all floorplans and MAC shares (`out/beta.json` floorplan_mac) |
| F05 | MAC location | column-decoder strip of each bank |  | [AttAcc](https://doi.org/10.1145/3620665.3640422) Sec. 4.1; [HBM-PIM](https://doi.org/10.1109/ISCA52012.2021.00013) Sec. III-A | verified |  |
| F06 | Stripe orientation | parallel to the GPU-facing edge |  | HotSpot-HMS model files ([Hong et al.](https://arxiv.org/abs/2403.09358)) (TSV stripe along y) | assumption | follows the HotSpot-HMS TSV stripe; perpendicular: beta 1.127 against 1.088 (x = 0, MAC 3.5 %; `out/beta.json` floorplan_mac a50_perp:e0.035 vs a50_par:e0.035); f_bank 1.1456 against 1.1146 (comparison only) |
| F07 | Half and quadrant subsets | half = one channel (32 cells, 64 banks); quadrant = one PC (16 cells, 32 banks) |  | JESD238B.01 Table 4 (32 banks per pseudo channel) | verified, note | a whole pseudo channel on one die is our choice |
| F08 | GPU-side base-die heat variant | half base die (4.0 x 12.19 mm): beta 0.669 at x 0.357 |  | [FGDRAM](https://doi.org/10.1145/3123939.3124545) Sec. 3.5 p.8; HotSpot-HMS model files ([Hong et al.](https://arxiv.org/abs/2403.09358)) (BD_PHY area); `out/recal.json` solution.cell_basephy | verified, note | the half-die block is larger than an HBM3 PHY, so PHY heat is less concentrated than in reality |

## HotSpot settings and calibration

| ID | Parameter | Value | Unit | Source (location) | Status | Note |
|---|---|---|---|---|---|---|
| P01 | GPU power map | non-memory power uniform over the GPU die block |  | `model/power.py` source_row("gpu"); `out/recal.json` best_fit.residual.gpu_die | assumption | no public GH100 power map; GPU-side variants: matched beta 0.652-0.671 (`checks/variants/package_variants.json`) |
| P02 | Measured power split P_G / P_M | P_G = power.draw - GPU Memory Power; P_M = GPU Memory Power | W | `data/SOURCES.md` (nvidia-smi power.draw, "GPU Memory Power" of -q -d POWER) | verified, note | memory power is NVIDIA's memory-subsystem average; absorbed by the fitted x, k, r_convec |
| P03 | Memory power distribution | (1 - x) area-weighted over all eight DRAM dies of every stack |  | `model/power.py` SEL["uni"]; `data/inputs/dram_map_decode.json` (uniform by construction) | assumption | channel interleaving spreads traffic; beta over ratio 0.93-0.96 across the refitted variants (`checks/variants/package_variants.json`) |
| P04 | Base-die share x (fitted) | 0.975 (grid 64), 0.901 (grid 128) |  | `out/recal.json` best_fit; `checks/numerics/out_best_xscan_g64_g128.json` | verified, note | x is not a robust physical value; its beta is |
| P05 | MAC share of PIM heat at column decoders | e = 3.5 % (0.16 / 4.56); range 0.9-30 % |  | [AttAcc simulator config.py:58](https://github.com/scale-snu/attacc_simulator/blob/c60005143a6b492d7ef83231723386478b59a506/src/config.py#L58); `model/power.py` E_TABLE; `out/bank.json` range_over | verified, note | at the paper's in-bank energies the share is 0.6-1.1 %, covered by the 0.9 % case (f +0.3 %) |
| P06 | Row-miss extra power placement | in the bank arrays (same distribution as PIM heat) |  | `steps/recal.py` model_fit() (row-miss extra power in "array") | assumption | also the PIM placement; beta over ratio 0.93-0.96 (`checks/variants/package_variants.json`) |
| P07 | DRAM background map total | 23.899 W per stack (bank.py constant 23.9022) | W | `data/inputs/dram_map_decode.json` stack_W; `steps/bank.py` BASE_DRAM_W (comment) | verified, note | two values for one map (+0.015 %); effect on f below 1e-4 |
| P08 | Idle memory power in the row-miss replay | m0 43.26 | W | `data/measured/h100_rowmiss_rows.csv` (3 idle rows); `out/recal.json` measured_points | verified |  |
| P09 | Bank-model base point and finite-difference width | GPU 80 W, DRAM 23.9 W/stack, PIM 40 W/stack; dM 20 | W | `steps/bank.py`, `steps/recal.py` betas(); `checks/numerics/out_e_readout_fd_g64.json` | verified, note | dM 0.5-20 W: beta within 0.4 %; 40 W: +2.2 % |
| P10 | beta operating point | H100 GPU+PIM at the threshold with no power constraint that beta gives: 1.872 pJ/bit at x = 0, 3.04 at the matched x; iteration starts at 1.84 pJ/bit | pJ/bit | `model/beta.py` fixed_point(); `data/inputs/beta_operating_points.json` (analysis repository cc8fdf6, abs_threshold.point); start point `data/inputs/beta_operating_point.json` | verified, note | fixed point, 3-6 iterations from 1.84 pJ/bit (P_G 332.7 W, P_M 80.1 W, P_P 52.1 W; the H100 threshold at the f_bank value 1.11, `data/inputs/beta_operating_point.json`); the exported points reproduce the H100 / H200 / B200 thresholds at beta 1.11 (1.836 / 1.529 / 1.899 pJ/bit). Held at 1.84 pJ/bit: x = 0 1.088, matched 0.643 (against 0.674). H200 / B200 operating points: x = 0 1.090 / 1.091 (direct), matched 0.683 / 0.688; one value per split serves all three GPUs. Analysis repository, not included: the decode-step and power model behind the operating points, fitted to our H100, H200 and B200 measurements (dram M04-M11) |
| P11 | HotSpot output resolution | 0.01 K (%.2f) | K | HotSpot temperature_grid.c:1312, 1323; `checks/numerics/out_superpos.json` | verified, note | 0.01 K over 200 W per stack = 5e-5 K/W; superposition error 0.009 K |
| P12 | Step-2 calibration endpoints and memory share | 80 / 480 W GPU+step power on the DRAM map; memory share 0.1175 | W | `steps/calibrate.py`; `out/calibrate.json` | verified, note | Linear: endpoint values do not matter |
| P13 | Subset placement cases | half 1.51-1.80, quarter 1.75-2.86 (x = 0, a50 / a75 par) |  | `model/power.py` SUBSETS; `out/beta.json` summary.half, quarter | verified, note | Bounding cases for PIM on a subset of banks; f_bank normalisation (comparison only; k = 1, grid 128): 1.50-1.85 / 1.75-3.04 |
| P14 | Ripple phases | step 17.97 ms; PIM on 1.70 ms at 133.9 W/stack; off: GPU 359.1 W, memory 15.35 W/stack |  | `data/inputs/ripple_phases.json` (derivation inside); `data/inputs/gpu_side_calibration.json` (mu_K, s, rail_heat) | verified, note | in-bank 2.24 pJ/bit, above the x = 0 beta point (1.87) and below the matched one (3.04). Analysis repository, not included: the decode-step power model (inputs in `data/inputs/gpu_side_calibration.json`) |
| S01 | HotSpot version | HotSpot 6.0 (47b1e83c19) + layer-dump.patch (dump only) |  | [HotSpot](https://github.com/uvahotspot/HotSpot/tree/47b1e83c19) 47b1e83; `hotspot/layer-dump.patch` | verified, note | the patch adds an all-layer grid dump; solver untouched |
| S02 | Model type | grid + layer file + detailed_3D on |  | [HotSpot 6.0 report](https://www.cs.virginia.edu/~skadron/Papers/HotSpot60_TR.pdf) Sec. 3; README-6.0 | verified |  |
| S03 | Steady solver and tolerance | iterative (SUPERLU = 0), stop when max update < 1e-6 | K | HotSpot 6.0 report Sec. 4; `hotspot/Makefile` SUPERLU=0; `checks/numerics/tol_summary.json` | verified, note | the report recommends the direct solver for heterogeneous layers; 1e-6 vs 1e-10: 0.01 K (GPU source) |
| S04 | Grid | 64 x 64 (fits), 128 x 128 (full mode: bank model, beta at x = 0 1.082) |  | `checks/numerics/out_conv_x0.json`, `out_best_xscan_g64_g128.json` | verified, note | grid 64 -> 128: beta(x=0) -0.7 %, matched beta +0.3 % |
| S05 | Detailed-3D lateral conductance asymmetry | energy balance +2.3 to +2.8 | % | HotSpot temperature_grid.c:2302-2312; `checks/numerics/eb_bank.json`, `out_sym_refit.json` | verified, note | HotSpot 6.0 behaviour; symmetric build: matched beta within 0.25 % |
| S06 | Ambient and initial temperature | 50 | C | [Hong et al.](https://arxiv.org/abs/2403.09358) Table II; `geometry/h100_5stack.config` ambient | verified |  |
| S07 | Linearity | leakage off, fixed r_convec, no secondary path |  | [Huang et al.](https://doi.org/10.1109/TVLSI.2006.876103) Sec. I; `checks/numerics/out_superpos.json` | verified | chip leakage is inside the measured powers |
| S08 | HotSpot capacitance factor | C_FACTOR 0.333 |  | HotSpot temperature.h:114; `checks/validation/trans_cmp_match.json`, `meas_t63.json`, `e3_ripple_c.json` | verified, note | model HBM 63 % at 0.46 s, measured 14-18 s; transient only |
| S09 | Transient step | rows every 0.1 ms; adaptive RK4 starting at 1e-7 | s | HotSpot temperature.h:117, temperature_grid.c:3225-3236; `out/bank.json` half_window_max_diff | verified, note | Grid 128 vs 64 step response within 2 % after 0.5 ms |
| S10 | Modal fit of step responses | 36 time constants 0.2 ms-20 s; ridge 1e-3 |  | `model/modal.py`; `out/bank.json` modal_fit_residual_rms_max_K | verified, note | residual rms 0.003 K = rounding noise of the 0.01 K output; max 0.012 K |
| S11 | Temperature readout | hottest grid cell of the DRAM dies (sensor assumed at the hottest cell) |  | JESD238B.01 Sec. 13.5.12, Table 121 (no placement); `model/readout.py`; `out/recal.json` die_average | verified, note | die-average sensors (bottom die / base die) in the fit: matched beta 0.748 / 0.807, computed here; hottest cell 0.14-0.36 mm from the die edge (`checks/numerics/out_peak2_g*.json`); coarse readout (2 x 2 block centre, step 2): x = 0 1.023 against 1.128 (-9 %), matched 0.655 against 0.659 (local slope, `checks/numerics/out_e_readout_fd_g64.json`) |
| S12 | Block temperature mapping | grid_map_mode center |  | HotSpot HOWTO:147-151 | verified | step 2 only; step 4 refits r |
| S13 | Inert config entries | thermal_threshold, dtm, base_proc_freq, floorplanner, interface, secondary path |  | `geometry/h100_5stack.config` | verified, note | Entries that never enter the grid solve, or are overwritten per run |
| S14 | Sink-resistance grid and interpolation | r grid 0.008-0.020 K/W; quadratic in r | K/W | `steps/recal.py` RS, interp_r(); `checks/numerics/out_step1_direct_vs_interp.json` | verified, note | interpolation error <= 0.3 %; GPU-side fits extrapolate up to 0.0002 K/W below the grid (quick) |
| S15 | Bond-k grid | k 0.75-2.0 (quick), -1..3 x scan step 0.05 |  | `steps/recal.py` KS, solve_x(); `out/recal.json` | verified |  |
| S16 | Step-2 secant stop rule | \|slope/target - 1\| < 0.2 % (achieved -0.12 %) |  | `steps/calibrate.py`; `out/calibrate.json` k.k1.slope_check | verified |  |
| S17 | beta definition | one definition: memory power with the same peak DRAM rise as the PIM power, over the PIM power, at the H100 threshold point it gives; 1.088 at x = 0 |  | `model/beta.py`; `steps/beta.py`, `steps/recal.py` eqv_betas(); definition test in the analysis repository (not included) | verified, note | comparison normalisations, kept as labelled fields and not used: (f_bank 1.115 at k = 1 and a fixed base point, `out/bank.json` f_bank); local slope 1.128 and regression slope at a fixed operating point (`*_local`). Test against direct HotSpot thresholds of the calibrated package (every split x in 0-1, every bank and channel subsets, three GPUs): this definition puts the threshold within -1.8 to +0.6 % (6e-5 with the GPU power held); the local slope misses by up to -30 %, f_bank by up to -46 % |
| S18 | Unit amplitudes and step-2 k grid | GPU 1000 W; other sources 200 W per stack; step-2 k = 1 (default), `--all-k` adds 1.5-16 |  | `model/power.py` GPU_W, UNIT_W; `steps/calibrate.py` KS_ALL | assumption | the model is linear, so amplitudes only keep the response above HotSpot's 0.01 K output step (P11); other k are not used downstream |
| S19 | Bootstrap and row selection | 2000 draws; seed 0 (block bootstrap over files), 20261110 (row-miss pairs); rows with settled = 1 (and valid = 1 for the row-miss run) |  | `model/measured.py` h100_fit(), rowmiss(); `data/SOURCES.md` (column meanings from the measurement kit) | assumption | settled and valid flags are set by the measurement kit; the CI is resampling only |

## Measured and fitted inputs

| ID | Parameter | Value | Unit | Source (location) | Status | Note |
|---|---|---|---|---|---|---|
| M01 | HBM temperature rise per non-memory W a_G | 0.02259 [0.0193, 0.0258] | K/W | `data/measured/h100_exp1/*.csv` (7 files); `steps/measured.py`, `out/measured.json` h100_fit.s_host | verified | OLS on 89 settled rows; 95 % CI by block bootstrap over files |
| M02 | HBM temperature rise per memory W a_M | 0.18774 [0.168, 0.206] | K/W | same regression, `out/measured.json` h100_fit.s_rail | verified | nvidia-smi HBM temperature; JESD238B.01 Sec. 13.5.12 does not place the sensor (S11) |
| M03 | Intercept, residual, highest temperature in the fit | c 29.87 C, residual sd 1.99 C, highest measured 72.0 | C | `out/measured.json` h100_fit | verified, note | Linear model without leakage: offset affects absolute maps only, not slopes or beta |
| M04 | GPU-die temperature rise per non-memory W | 0.0655 [0.0633, 0.0690] | K/W | same regression on tgpu; `out/recal.json` checks_measured | verified | check, not fitted: model 0.0556, -15 % |
| M05 | Anchor slope and memory share (GEMM duty sweep) | 0.04328 K per board W; 0.1175 memory W per board W |  | `data/measured/h100_exp1/h100_anchor_r*.csv` (60 settled GEMM rows); `out/measured.json` anchor | verified |  |
| M06 | Row-miss heating ratio | 0.701 [0.633, 0.779] |  | `data/measured/h100_rowmiss_rows.csv` (38 valid, settled rows); `out/measured.json` rowmiss | verified, note | target for x; pairs bootstrap CI; bytes are logical counts |
| M07 | Heat-sink resistance r_convec (k = 1 model) | 0.01361 | K/W | `steps/calibrate.py`; `out/calibrate.json` k.k1 | verified, note | target 0.04328 K/W; secant extrapolation from r 0.015 / 0.025, check run -0.12 %; r_convec 0.1-0.2 cannot match the measured slopes (a_G 5-10 times), larger spreader/sink refits give r 0.004-0.006 and beta within 0.3 % (`checks/rconvec/`) |
| M08 | Bond layer conductivity in steps 2-3 (calibration, bank model) | k = 1.0 (literature 1.5) | W/m-K | [Xylem](https://iacoma.cs.uiuc.edu/iacoma-papers/micro17_2.pdf) Sec. 2.5 p.3 (about 1.5 W/m-K, 20 µm); [Hong et al.](https://arxiv.org/abs/2403.09358) Table II (bonding 1.5) | assumption | modelling choice, conservative: close to the x = 0 fit (k 0.954, row-miss ratio 1.18, above the CI), which gives the highest beta (1.088 against 0.668-0.736 for x in the CI, one beta definition); the joint fits span k 0.95-1.95 (G20) |
| M09 | Joint fit at x = 0 | k 0.954, r 0.01102, beta 1.088, row-miss ratio 1.18 |  | `out/recal.json` summary.x0 | verified | Targets a_G, a_M; residual ~0 |
| M10 | Joint fit at x = 1 | k 1.952, r 0.01030, beta 0.668, ratio 0.693 |  | `out/recal.json` summary.x1 | verified | Targets a_G, a_M; residual ~0 |
| M11 | Best fit (x matched to row-miss ratio) | x 0.975, k 1.927, r 0.01031, beta 0.674; GPU-die slope -15 % |  | `out/recal.json` best_fit | verified, note | three targets, three parameters, residuals about 0 (not a validation); x moves 0.36-0.98 with grid, base-die placement and readout while beta stays 0.67 for the cell readout |
| M12 | Allowed range (ratio inside CI) | x 0.74-1.0, beta 0.67-0.74 |  | `out/recal.json` allowed; second card `checks/validation/e2_card2.json` | verified, note | second H100 card: 0.736 at x = 1 (local slope) |
| M13 | GPU-side base-die variant fit | x 0.357, k 1.44, r 0.0079, beta 0.669 |  | `out/recal.json` summary.basephy_target; [FGDRAM](https://doi.org/10.1145/3123939.3124545) Sec. 3.5 p.8 (PHY stripe) | verified, note | r 0.0079 below the r grid (0.008): quadratic extrapolation, <= 0.3 % (`checks/numerics/out_step1_direct_vs_interp.json`) |
| M14 | Base-die share from our HBM3E split (reference only) | x 0.072 / 0.165 / 0.234; prior beta 0.94-1.04 |  | HBM-Power main 61aada9 `config/HBM3E_6400MTs/power.json` with v0.3-alpha timing and organisation; `data/inputs/hbm3e_read_split.json` | verified, note | our I/O-and-TSV lower bound, not a published value; not used in the paper; labelled reference only in README, figure and reference.json |
| M15 | Bank heat factor f_bank (comparison only) | 1.1146 (range 1.0251-1.1513); quick mode 1.1205 (1.0234-1.1548) |  | `steps/bank.py`; `out/bank.json` summary.f_bank | verified, note | not the model's beta (S17): k = 1, fixed base point (GPU 80 W), normalised by uniform DRAM-die power of the same amplitude; beta at x = 0 is 1.088 (1.026-1.132), `out/beta.json`. Grid 128 in this repository (`python3 -m thermal.steps.bank --grids 64,128`, cached runs): 1.1145 (1.0250-1.1512) against 1.1146 there (unit-response superposition, about 1e-4). f_bank and the local slope at x = 0 (1.128, grid 64) differ by 0.7 % on the same grid through k (1 against 0.954), r_convec (0.01361 against 0.01102), the base point and the normaliser |
| M16 | Base-die vs DRAM-die heating per W | 1.74 (k = 1); 1.63 (best fit) |  | `out/bank.json` summary.base_die_over_dram; [Eckert et al.](https://www-old.cs.utah.edu/wondp/eckert.pdf) p.4 | verified, note | consequence of bond k 1-2 W/m-K; Eckert et al. find about 1:1 for their cooling solutions |
| M17 | Per-layer PIM ripple | max 0.055 K above the step average (bound 0.1 K); burst bound 0.14 K; lumped 3.7 K if all PIM time of a step is one block | K | `out/bank.json` ripple; `checks/validation/e3_ripple_c.json` | verified, note | HotSpot capacitance is 1/3 of physical (C_FACTOR); with 3x the ripple is 0.018 K, so the bound is conservative |
| M18 | Measured HBM2 base-die + external share | about 49 % of HBM2 read power | % | [HBM-PIM](https://doi.org/10.1109/ISCA52012.2021.00013) Fig. 11 (read from the figure) | verified, note | reference for x only; Hong et al.'s 10 W base-die power is a PIM-logic TDP, not an HBM share |
| M19 | Uniform PIM peak rise | 0.93 K per W per stack (k = 1) | K/W | `out/calibrate.json` k.k1.pim_peak_per_w_per_stack | verified |  |
## Sources

- JEDEC, High Bandwidth Memory (HBM3) DRAM, JESD238B.01, 2025 (cited by section and table only).
- NVIDIA H100 Tensor Core GPU Architecture whitepaper V1.02, https://resources.nvidia.com/en-us-tensor-core/gtc22-whitepaper-hopper
- J. Hong et al., Bandwidth-Effective DRAM Cache for GPUs with Storage-Class Memory, HPCA 2024, https://arxiv.org/abs/2403.09358 (Table II; the HotSpot-HMS model files are the source of the package setup).
- A. Agrawal, J. Torrellas, S. Idgunji, Xylem, MICRO 2017, https://iacoma.cs.uiuc.edu/iacoma-papers/micro17_2.pdf
- S. Lee et al., Hardware Architecture and Software Stack for PIM Based on Commercial DRAM Technology (HBM-PIM), ISCA 2021, https://doi.org/10.1109/ISCA52012.2021.00013
- M. O'Connor et al., Fine-Grained DRAM (FGDRAM), MICRO 2017, https://doi.org/10.1145/3123939.3124545
- J. Park et al., AttAcc!, ASPLOS 2024, https://doi.org/10.1145/3620665.3640422; simulator https://github.com/scale-snu/attacc_simulator at c600051
- Y. Eckert, N. Jayasena, G. Loh, Thermal Feasibility of Die-Stacked Processing in Memory, WoNDP 2014, https://www-old.cs.utah.edu/wondp/eckert.pdf
- A. Olgun et al., Understanding the Power Consumption of Modern High Bandwidth Memory, arXiv:2610.14244; HBM-Power https://github.com/CMU-SAFARI/HBM-Power (v0.3-alpha 4642c61, main 61aada9)
- W. Huang et al., HotSpot, IEEE TVLSI 2006, https://doi.org/10.1109/TVLSI.2006.876103; R. Zhang, M. R. Stan, K. Skadron, HotSpot 6.0 report, UVA CS-2015-04, https://www.cs.virginia.edu/~skadron/Papers/HotSpot60_TR.pdf; code https://github.com/uvahotspot/HotSpot at 47b1e83
- CoMeT, https://github.com/marg-tools/CoMeT at 250ba95
- ServeTheHome, SK hynix HBM packaging at Hot Chips 2026, https://www.servethehome.com/sk-hynix-hbm-packaging-at-hot-chips-2026/
