# Heat-sink resistance r_convec

r_convec is HotSpot's single resistance from the heat sink to ambient (`hotspot/temperature_grid.c` sink layer,
`temperature.c` peripheral sink nodes, split by area). There is no per-die convection: inside the package heat moves by
conduction only. Every watt of the package leaves through r_convec, so the GPU rise per watt of non-memory power is at
least r_convec + 0.0026 K/W (sink conduction); the measured 0.0655 K/W bounds r_convec below about 0.063 K/W.

The fitted r_convec (about 0.011 K/W) is an effective sink-to-ambient resistance fitted to the measured H100 slopes. It
sits at HotSpot's own lower bound for the best cooling (`hotspot/package.h` R_CONVEC_LOW 0.01, liquid cooling) and is
about seven times lower per sink area than the 0.143 K/W of the HotSpot-HMS air-cooled setup (about 100 W on a 60 mm
sink, about 14 K over ambient; the H100 at 698 W gives about 8 K at 0.011 K/W).

**Sweep** (k 0.954, x = 0; measured a_G 0.0226, a_M 0.1877, GPU 0.0655 K/W; 698 W point: HBM 63 °C, GPU 66 °C)

| r_convec (K/W) | a_G | a_M | GPU slope | beta | HBM at 698 W | GPU at 698 W |
|---|---|---|---|---|---|---|
| 0.011 (fit) | 0.0225 | 0.187 | 0.057 | 1.088 | 63 °C | 61 °C |
| 0.029 | 0.044 | 0.206 | 0.081 | 1.075 | 78 °C | 77 °C |
| 0.1 | 0.119 | 0.280 | 0.158 | 1.051 | 130 °C | 131 °C |
| 0.143 | 0.164 | 0.324 | 0.203 | 1.044 | 161 °C | 162 °C |
| 0.2 | 0.222 | 0.383 | 0.262 | 1.037 | 202 °C | 204 °C |

With r_convec fixed at 0.1 or 0.143, no bond conductivity k in 0.3-4 and no base-die share x in 0-1 brings a_G below
0.115 or 0.160 K/W (5.1 and 7.1 times the measurement).

**Larger spreader and sink**, refitted to the same measurements (x = 0; checked by direct runs at the fitted point)

| spreader / sink (mm) | k | r_convec | beta | H100 threshold (pJ/bit) | GPU slope error |
|---|---|---|---|---|---|
| 45.6 / 82.1 (model) | 0.954 | 0.0110 | 1.088 | 1.872 | -13 % |
| 60 / 120 | 0.947 | 0.0059 | 1.085 | 1.878 | -12 % |
| 80 / 150 | 0.951 | 0.0038 | 1.086 | 1.877 | -12 % |
| 60 / 120, spreader 3 mm | 0.955 | 0.0058 | 1.087 | 1.875 | -14 % |

A larger spreader carries more GPU heat to the stacks, so the fit needs a smaller r_convec. beta and the threshold move
by less than 0.3 %.

**Direction of beta.** A larger r_convec adds the same rise per watt to every source, so beta moves toward 1 at every
split: 1.088 to 1.044 at x = 0 and 0.625 to 0.737 at the matched split (r 0.011 to 0.143). The beta of the model
(1.09, x = 0) stays above the beta of every split over r_convec 0.011-0.2.

**Files.** `summary.json` (the numbers above); sweep `out_abd.json`, `out_d.json` (`analyze_abd.py`, `analyze_d.py`,
`common.py`, `run_batch.py`); refits `verify/sol.json` and the scripts in `verify/`, package configs in `configs/`.
`out_c.json` (`fit_c.py`) is an earlier refit whose r was extrapolated below its r grid; `verify/` supersedes it.
The scripts are records: they ran on HotSpot runs in a scratch folder (`<scratch>`) with this repository's model code.
