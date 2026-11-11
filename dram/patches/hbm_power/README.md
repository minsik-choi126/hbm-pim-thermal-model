# HBM-Power patches

Changes to [HBM-Power](https://github.com/CMU-SAFARI/HBM-Power) at `HBM-Power.commit` (tag v0.3-alpha, Zenodo
10.5281/zenodo.21970747). HBM-Power has no top-level license, so only these diffs are kept here. The files they modify
come with their own licenses, copied next to the patches:

| patch | modifies | upstream license | copy |
|---|---|---|---|
| 01-json-per-bank-and-shared | `sources/drampower` (DRAMPower, HBM runners and report) | BSD 3-Clause (DRAMPower, TU Kaiserslautern / Fraunhofer IESE; HBM-Power's copy carries no license file) | `LICENSE.drampower` |
| 02-hbm3-refresh | `sources/drampower` (HBM3 refresh energy, `HBM3_runner`) | BSD 3-Clause, as above | `LICENSE.drampower` |
| 03-llmsim-portable-build | `sources/LLMSimulator` (build files; its bundled Ramulator 2 `param.h`) | MIT (SNU SCAL); bundled Ramulator 2: MIT (SAFARI) | `LICENSE.llmsimulator`, `LICENSE.llmsimulator-ramulator2` |

`LICENSE.drampower` is the upstream DRAMPower `LICENSE.txt` (github.com/tukl-msd/DRAMPower @a2068bc). The two MIT texts
are the files in HBM-Power at the pinned commit.

`dram/setup/hbm_power.sh --runner` applies 01 (without `HBM3_runner.cpp`, which 02 carries in its final form) and 02,
and builds `HBM3_runner`. Patch 03 is needed only to regenerate the command log of the bank power map with
LLMSimulator (`dram/bank_power/regenerate/`).
