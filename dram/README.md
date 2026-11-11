# dram/: DRAM and PIM side

PIM bandwidth, power and energy of AttAcc's bank-level PIM on H100, H200 and B200, the IDD7 stack power from public
HBM data, the HBM3E read-energy split that gives the base-die share prior (sensitivity only; the thermal fit sets x),
and the bank power map handed to HotSpot. Every parameter and its source: [`PARAMETERS.md`](PARAMETERS.md).

```
AttAcc Ramulator 2 (+ patches) -- mac_sweep, refresh_sweep, e2e --> PIM bandwidth per MAC interval, refresh slowdown
AttAcc energy table                                              --> PIM energy 4.56 pJ/B  --> PIM power per stack
HBM-Power (Olgun et al.) IDD data + HBM3E model                  --> IDD7 stack power; base-die share prior (sensitivity)
DRAMPower per-bank energy (H100 decode trace)                    --> bank power map --> thermal/ (ptrace)
```

## Run

```bash
bash dram/run.sh quick    # derived values from the committed simulator outputs into dram/build/quick, then the check
bash dram/run.sh full     # build AttAcc, rerun every simulation into dram/build/full, compare with dram/results
python3 -m dram.check     # check the committed dram/results only (from the repository root)
```

Needs Python 3 with numpy, git, and for `full` cmake >= 3.14 and a C++20 compiler (Apple clang or g++ >= 12).
`dram/setup/attacc.sh` makes its own venv with numpy and pandas for AttAcc's `main.py`.
External sources go to `dram/external/` (git-ignored). Tested on macOS.

## Values

| quantity | H100 / H200 / B200 | script |
|---|---|---|
| PIM bandwidth at the power-constrained MAC interval | 5.38 / 5.72 / 5.45 TB/s per stack | `attacc.mac_sweep` -> `attacc.pim_table` |
| PIM power there (4.56 pJ/B x bandwidth) | 24.5 / 26.1 / 24.8 W per stack | `attacc.pim_table` |
| PIM bandwidth at tCCD_L | 7.40 / 8.83 / 8.90 TB/s per stack | `attacc.pim_table` |
| MAC interval (simulator input) | 6 / 7 clocks: AttAcc presets; 9 clocks: our choice for 8.0 Gb/s (patch 02); 5-10 clocks give 8.9-5.0 TB/s per stack | `attacc.pim_table` |
| PIM energy | ACT 0.11 pJ/bit (FGDRAM Table 3) + cell 0.44 pJ/bit (AttAcc's own), MAC 0.32 pJ (0.16 pJ/B on K and V) | AttAcc `src/config.py` |
| KV-capacity batch, LLaMA-65B, context 2048 | 94 / 185 / 244 | `attacc.pim_table` |
| busiest channel / mean (kappa) | 1.064 / 1.038 / 1.049 | `attacc.pim_table` |
| slowdown at 2x refresh: r_gpu (analytic), r_pim (all-bank refresh) | r_gpu 1.11 / 1.13 / 1.13; r_pim 1.11-1.14 | `attacc.pim_table`; `attacc.refresh_sweep` |
| per-bank refresh, JESD238 timings, 2x / 1x | 1.157 / 1.168 / 1.169 (all-bank 1.109 / 1.135 / 1.133): slower than all-bank refresh; the paper uses all-bank refresh | `attacc.refresh_sweep` |
| AttAcc main.py, H100: PIM attention per layer | 31.0 us (model 31.2 us) | `attacc.e2e` |
| IDD7 stack power, 35 HBM2 chips | 1.10-1.51x AttAcc's PIM power (H100); highest chip 37.2 / 41.2 / 40.7 W | `hbm_power.idd7_chips` |
| base-die share x, I/O-and-TSV lower bound (our split of the HBM-Power HBM3E model) | 0.165 (0.072-0.234); measured HBM2 base die + external about 49 % (HBM-PIM Fig. 11) | `hbm_power.hbm3e_split` |
| DRAM floor per stack (bank power map) | 23.90 W, 32 blocks, spread < 0.001 % | `bank_power.stack_power` |

`results/reference.json` holds the 62 checked values with tolerances. The HBM2 PIM energy over AttAcc's table (1.35)
takes the read energy as IDD4R above IDD3N1, as DRAMPower and the IDD7 loop do.

## Setup choices

- **Data rates.** 5.2 / 6.4 / 8.0 Gb/s. H200 and B200 run at the 6.4 / 8.0 Gb/s bins, 2.4 % above the 6.25 / 7.81 Gb/s
  implied by datasheet bandwidth, so their PIM bandwidth is 2.4 % optimistic. HBM-Power also models H200 at 6.4 Gb/s.
- **Organisation.** All GPUs use the 8 Gb/channel organisation (`HBM3_8Gb_2R`). JESD238B.01 Table 4: 12 Gb/channel
  differs only in row count, and AttAcc has no 12 Gb 2-rank preset. tRFCab follows the real density (350 / 410 ns).
- **Capacity and model.** Datasheet decimal GB (80e9 / 141e9 / 180e9 B, conservative; H100's physical 85.9e9 B would
  give batch 103). Weights 65.0e9 x 2 B (LLaMA Table 2 gives 65.2B, which would give B200 batch 243).
- **Refresh.** Doubling above 85 C is DDR4 practice (Mukundan et al., ISCA'13 Sec. 7.3), not a JESD238 value; the
  threshold is defined at 85 C, no trip-point sensitivity run. 95 / 95 / 105 C are the NVML limits of the measured cards. Per-bank refresh
  (`refresh_sweep.py`): main mode `pbj` uses JESD238B.01 tRFCpb (Table 93, 200 / 240 ns by die density) and counts
  tRREFD per pseudo channel (Table 3); mode `pbh` (Ramulator 2's HBM2 tRFCsb scaled to 160 / 187 ns, tRREFD per SID)
  is kept for comparison and gives 1.085 / 1.095 / 1.093.
- **Energy.** 0.44 pJ/bit and the 1.01 / 1.23 datapath split are AttAcc's own (Sec. 7.1, datapath-length scaling);
  only the 0.11 pJ/bit ACT comes from FGDRAM Table 3. The in-bank share 0.44 / 3.18 leaves out the 0.3 interposer I/O,
  which errs toward more PIM heat (1.35x instead of 1.24x over 3.48). 0.32 pJ/MAC is AttAcc's table value with no
  cited source; the simulator charges it on the score layer only (0.08 pJ/B), we charge K and V (0.16 pJ/B, +1.8 %).
- **HBM-Power versions.** v0.3-alpha (Zenodo artifact) feeds the IDD7 model and the bank map. Main 61aada9
  `config/HBM3E_6400MTs/power.json` feeds the read split, with the v0.3-alpha timing and organisation files. The
  HBM3E IDD7 / PIM ratios 0.9-1.4x hold for v0.3-alpha; main's config gives 0.96x / 1.22x.
- **Not included.** Our Ramulator 2.1 port and its PIM runs: no number in the paper uses them.

## Layout

```
dram/
  run.sh  check.py  common.py  PARAMETERS.md
  setup/        attacc.sh (clone, patch, build), hbm_power.sh (sparse fetch, sha256 checks; --runner builds DRAMPower)
  patches/      attacc/ (00-03, pinned commits, licenses, README), hbm_power/ (01-03, pinned commit, licenses, README)
  attacc/       mac_sweep, refresh_sweep, refresh_cmd_check, concurrent_reads, e2e, pim_table
  hbm_power/    idd7_chips, hbm3e_split
  bank_power/   bank_power_to_ptrace (DRAMPower -> HotSpot ptrace), stack_power, lcf_columns,
                data/ (DRAMPower per-bank energy, 2 pseudo channels), regenerate/ (trace, converter, config)
  checks/       reruns behind single statements (REPEAT convergence, nCCDL, per-bank timings, DRAMPower data rate);
                outputs in checks/out/
  data/         measured_copy_power.json
  results/      committed outputs (CSV from the simulators, JSON from the scripts) and reference.json
```

## Runtimes

10-core Mac under heavy load (load average 30-90); an idle machine is faster.

| step | time |
|---|---|
| `setup/attacc.sh` (clone, patch, build) | 81 s (+15 s venv) |
| `setup/hbm_power.sh` | 2 s; `--runner` about 2 min |
| quick: pim_table, idd7_chips, hbm3e_split, stack_power, check | 2 s |
| full, end to end | about 33 min |
| mac_sweep, 5.2 / 6.4 / 8.0 Gb/s | 43 / 33 / 38 s |
| e2e (context 2048; `--all` adds AttAcc's default 128 output tokens, about 25 min) | 8 s |
| refresh_sweep (266 runs) | about 17 min |
| refresh_cmd_check (30 configurations, every command checked) | 16 min |
| concurrent_reads | 82 s |
| checks: repeat_check / rep4 / nccdl_check / jedec_pb_check / drampower_rate_check | 3 / 1 / 1 / 2 min / 5 s |

## Upstream

| tool | commit | license | in this repository |
|---|---|---|---|
| [AttAcc simulator](https://github.com/scale-snu/attacc_simulator) | c6000514 | MIT | patches only |
| [Ramulator 2](https://github.com/CMU-SAFARI/ramulator2) (AttAcc submodule) | b7c70275 | MIT | patches only |
| [HBM-Power](https://github.com/CMU-SAFARI/HBM-Power) v0.3-alpha (Zenodo 10.5281/zenodo.21970747) | 4642c61f | no top-level license | patches only; data fetched, sha256 checked |
| DRAMPower (HBM-Power fork, `sources/drampower`) | same | BSD 3-Clause (upstream DRAMPower) | patches 01-02; `patches/hbm_power/LICENSE.drampower` |
| LLMSimulator (HBM-Power, `sources/LLMSimulator`) | same | MIT (SNU SCAL); bundled Ramulator 2 MIT | patch 03; `patches/hbm_power/LICENSE.llmsimulator*` |
| HBM-Power main, `config/HBM3E_6400MTs/power.json` | 61aada9a | as HBM-Power | fetched, sha256 checked |

Full hashes are in `patches/*/*.commit` and `setup/hbm_power.sh`. Patch details: `patches/attacc/README.md`,
`patches/hbm_power/README.md`.

## Inputs and provenance

- `results/*.csv`: AttAcc Ramulator 2 and `main.py` outputs; `full` regenerates them and `check` compares byte for byte
  (the e2e CSV row by row). `refresh_cmd_check` checks every issued command of the refresh runs.
- `data/measured_copy_power.json`: memory power the GPU reports (nvidia-smi) with a full-duty copy kernel and at idle,
  per stack, mean over the settled rows of our runs. Its `source` field gives the commit (38e0926 of our measurement
  repository), files and row counts. H200 is one of three measured cards, the middle value (others -1.3 % / +1.6 %).
  It scales the HBM2 IDD7 ratios. The copy kernel reads and writes with activations, so r x copy is upper-side.
- `bank_power/data/*.json`: DRAMPower (HBM-Power fork, patches 01-02) per-bank energy of one H100 channel, Llama-3.1-8B
  decode, batch 1, context 1024. Power while busy (79 % of the trace span). Data toggle 0.373 measured on
  Llama-3.1-8B up_proj weights (33.5 MB sample, sha256 fa04f8bb..., not included); TSV = DQ and bank group 0 are
  assumed. To regenerate: LLMSimulator (HBM-Power, patch 03) with `regenerate/llmsim_config_*.yaml` writes the
  command log (included, gzip); `convert_hbm3_log.py` turns it into per-pseudo-channel traces; `HBM3_runner <org>
  <timing> <power> <trace> --dq-rate=0.373 --tsv-rate=0.373 --bg-rate=0 --json=out.json` with the figure16 HBM3
  configs. The current patched runner gives 0.03 % less than the committed files (precharge background accounting);
  the committed files are the model's input.
- **Data rate of the bank map.** The trace comes from Ramulator 2 at HBM3 5.2 Gb/s (tCK 0.769 ns) but was priced with
  HBM-Power's 6.4 Gb/s timing file (tCK 625 ps), so its 105,305 cycles span 65.8 us instead of 81.0 us. Repricing at
  5.2 Gb/s (same ns timings and currents) gives 23.80 W per stack instead of 23.90 W, -0.4 %: per-command energies
  follow the ns timings, and background current dominates. `checks/drampower_rate_check.py`, output
  `checks/out/drampower_rate_check.json`. The map is a uniform background, so this moves only the base point.

## References

- J. Park et al., AttAcc! Unleashing the Power of PIM for Batched Transformer-based Generative Model Inference,
  ASPLOS 2024, doi:10.1145/3620665.3640422.
- H. Luo et al., Ramulator 2.0: A Modern, Modular, and Extensible DRAM Simulator, IEEE CAL 2023.
- A. Olgun et al., Understanding the Power Consumption of Modern High Bandwidth Memory: Experimental Analysis and
  Modeling Using Real HBM2 DRAM Chips, MICRO 2026 (arXiv:2610.14244); artifact v0.3-alpha, doi:10.5281/zenodo.21970747.
- S. Lee et al., Hardware Architecture and Software Stack for PIM Based on Commercial DRAM Technology, ISCA 2021.
- M. O'Connor et al., Fine-Grained DRAM: Energy-Efficient DRAM for Extreme Bandwidth Systems, MICRO 2017.
- J. Mukundan et al., Understanding and Mitigating Refresh Overheads in High-Density DDR4 DRAM Systems, ISCA 2013.
- JEDEC JESD238B.01, High Bandwidth Memory (HBM3) DRAM (cited by section and table).
