# Parameters and their sources

Every power value, parameter, hyperparameter and modelling choice of the model is listed with its source in two tables:

- [dram/PARAMETERS.md](dram/PARAMETERS.md): 66 entries. HBM organisation and timing, refresh, PIM energy and power,
  IDD7 data, bank power map, GPU-side measured inputs.
- [thermal/PARAMETERS.md](thermal/PARAMETERS.md): 85 entries. Package geometry and materials, DRAM-die floorplan,
  HotSpot settings, calibration, measured and fitted values.

Each entry was checked twice against its primary source: paper section, page, table or figure; JESD238B.01 by section
and table number; upstream code at a pinned commit by file and line; our measurement files. Derived values were
recomputed, and simulator or HotSpot settings were rerun where the note says so. `make check` runs
[`tools/check_params.py`](tools/check_params.py), which compares 295 printed values in these tables and in the README
with the code, config and data files the model reads.

## Reading the Status column

| status | meaning | where the evidence is |
|---|---|---|
| verified | value and unit equal the source; derived values recompute | Source column |
| verified, note | value confirmed; a caveat applies | Note column |
| assumption | no direct source; a stated choice | Note column gives the reason and the effect of the alternative where one was run, else "no sensitivity run" |
| documented | a known deviation kept on purpose | Note column and the file that sets it |

Two kinds of entry carry extra evidence in the Note column. Fitted values (thermal M07, M09-M13, G20; dram M06, M08,
M09, M11) give the target, the method and the residual. Numerical settings (thermal S03, S04, S09, S10, S14-S16, P09,
P11; dram D25) give the convergence test: grid 64 against 128, solver tolerance, finite-difference width, interpolation
against direct runs, REPEAT 4 / 8 / 16.

| table | verified | verified, note | assumption | documented |
|---|---|---|---|---|
| dram | 17 | 44 | 5 | 0 |
| thermal | 17 | 50 | 17 | 1 |

## Assumptions that move β most

β is the memory power that raises the peak DRAM temperature (hottest DRAM grid cell) as much as the PIM power does,
divided by the PIM power, at the H100 GPU+PIM threshold point that this β gives. Matched β is the value at the base-die
share x that reproduces the measured row-miss heating ratio: 0.674 (0.668-0.736 over the 95 % CI of the ratio). Rows
whose evidence is in `thermal/checks/` were run with the local-slope normalisation (comparison only, thermal S17) and
give its change. No sensitivity run: stack footprint, dark-site position (thermal G04-G06), 85 °C trip temperature.

| assumption (matched fit) | alternative run | matched β or effect | evidence |
|---|---|---|---|
| memory power split between DRAM dies and base die by x, fitted | x = 0 (all in the DRAM dies) / x = 1 | 1.088 / 0.668 | `thermal/results/reference.json` (recal) |
| HBM temperature = hottest DRAM grid cell | die-average sensor on the bottom DRAM die / on the base die | 0.748 / 0.807 | same |
| base-die heat over the whole base die | on the GPU-side half | 0.669 | same |
| slopes from one H100 card | second H100 card | 0.736 at x = 1 (local slope; this card 0.654) | `thermal/checks/validation/e2_card2.json` |
| operating point at the H100 threshold this β gives (fixed point) | fixed 1.84 pJ/bit | 0.674 against 0.643 | `thermal/results/reference.json` (recal) |
| spreader k 400, 1 mm | vapor chamber, k 4000, 3 mm | +0.018 to +0.039 | `thermal/checks/geometry/summary_result.json` |
| top TIM 20 µm, 5 mm²K/W | 56 µm everywhere / 14 mm²K/W over the HBM | 0.655 / 0.676 (local slope; base 0.664) | same |
| HotSpot-HMS layer thicknesses | 100 µm interposer or base die, 35 µm DRAM dies, 135 µm top die | -0.007 to +0.005 | same |
| grid 64 in the fit | grid 128 | +0.3 % | `thermal/checks/numerics/out_best_xscan_g64_g128.json` |
| PIM heat in every bank, MAC share 3.5 % | bank-array area 75 %, stripe orientation, MAC share 0.9-30 % | β 1.03-1.13 at x = 0 | `thermal/results/reference.json` (beta) |

## Inputs that set the threshold

The threshold is the in-bank PIM energy (pJ/bit) at which the HBM reaches 85 °C. It is computed outside this repository
from β and the GPU-side inputs below; their effects are those measured or recomputed there.

| input | value | alternative and effect | entry |
|---|---|---|---|
| doubled-refresh trip temperature | 85 °C | threshold defined at 85 °C; JESD238 leaves the trip point to the vendor; no sensitivity run | dram D17 |
| in-bank share of read energy | 26.35 % (HBM-PIM) | 15.3 % (AttAcc) / 45.1 % (FGDRAM) / 77.7 % (all but I/O) | dram E07, E08 |
| memory energy per KV byte mu_K (H200 / B200) | 63.06 / 44.19 pJ/B | -17 % with the same-card reference; +3.9 % for B200 at 7.7 TB/s | dram M07 |
| HBM heat share of dynamic memory power | 0.832 (H100) | other H100 cards 0.93-0.94 | dram M09 |
| DRAM background power in the bank map | 23.899 W/stack | 5.2 Gb/s DRAMPower timing: -0.43 % | dram E19 |
