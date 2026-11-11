# AttAcc simulator patches

Changes to the [AttAcc simulator](https://github.com/scale-snu/attacc_simulator) (MIT, `LICENSE.attacc`) and its
Ramulator 2 submodule (MIT, `LICENSE.ramulator2`). Pinned commits: `AttAcc.commit`, and `RAMULATOR2.commit`, the
Ramulator 2 commit AttAcc's `set_pim_ramulator.sh` resets the submodule to. The patches are `git diff` output against
the tree that script leaves, applied inside `ramulator2/` with `git apply -p2`; `dram/setup/attacc.sh` does all of it.
Some upstream files have CRLF line ends, which the patches keep (`.gitattributes`: `-text`).

| patch | needed for | effect on results |
|---|---|---|
| 00-macos-build | Apple clang only | none |
| 01-hbm3-pim-per-bank-refresh | per-bank refresh runs; fixes REFab timing | all-bank refresh 1x +0.2-0.3 %, 2x +0.4-1.0 % of runtime |
| 02-hbm3e-8gbps-preset | B200 (8.0 Gb/s) presets | new presets only |
| 03-concurrent-reads | concurrent_reads.py (PIMDualTrace, options) | none at the defaults |

## What `00-macos-build.patch` changes

Two build fixes for Apple clang (`FMT_CONSTEVAL=` for the fmt bundled in spdlog, `.template as<T>()` in
`src/base/param.h`). No effect on simulation results.

## What `01-hbm3-pim-per-bank-refresh.patch` changes

**Per-bank refresh for HBM3-PIM** (`src/dram_controller/impl/refresh/per_bank_refresh_hbm3.cpp`, new; registered in
`src/dram_controller/CMakeLists.txt`). `RefreshManager: impl: PerBankHBM3` sends one `REFsb` to every bank of every
pseudo channel per tREFI (`nREFI`; a 2x-refresh run halves it in the config).

- `policy: batched` (default): all banks back to back at each tREFI, alternating SIDs. This is the best case for
  AttAcc's all-bank PIM, which needs every bank open.
- `policy: spread`: one bank per pseudo channel every tREFI / banks. With tRFCsb longer than that interval an all-bank
  PIM can never re-open the channel, so this configuration is refused at setup.
- tRFCsb is `nRFCSB` (the HBM3-PIM presets give 260 / 320 cycles = 200 ns at 5.2 / 6.4 Gbps, independent of density;
  `refresh_sweep.py` sets it per die density: JESD238B.01 Table 93 tRFCpb 200 / 240 ns).

**Timing rules** (`src/dram/impl/HBM3-PIM.cpp`):

- channel: `REFsb -> ACTAB/ACTSB/ACTPB` >= tRFCsb (a bank under refresh cannot be re-opened by an all-bank, same-bank or
  per-bank activate); `PREA -> REFsb` >= tRP; `MACAB -> PRE` >= tRTP(L); `ACTAB -> PRE` >= tRAS
- bank: `REFsb -> ACT/REFsb` >= tRFCsb; `PRE -> REFsb` >= tRP
- `REFsb` prerequisite: close the channel (`PREA`), as `REFab` does
- option `refsb_rrefd_level: pseudochannel` (DRAM config, default `rank`): count tRREFD between `REFsb` to different
  banks per pseudo channel instead of per SID (JESD238B.01 Table 3 counts tRREFD per pseudo channel; the main per-bank
  mode `pbj` of `dram/attacc/refresh_sweep.py` uses it)

**Fixes that also affect the original runs** (measured against the original build: no refresh −0.11 to 0 %, all-bank
refresh 1x +0.2–0.3 %, 2x +0.4–1.0 % of runtime):

- channel-level `PREA -> REFab` >= tRP and `REFab -> ACTAB/ACTSB/ACTPB` >= tRFC. `REFab` is a channel-scope command,
  so the timing check never descends to the pseudo-channel nodes and the original pseudo-channel rules ending in `REFab`
  were never applied: a `REFab` followed a `PREA` after 1 cycle, and an all-bank activate re-opened the channel one
  cycle before the other pseudo channel's refresh ended (both seen in command logs).
- `hbm3_pim_controller.cpp`: the second command of a cycle was chosen with `req_it->command` after `req_it` could have
  been removed from its buffer (undefined behaviour). The first command is now saved before removal, and nothing is
  dual-issued when nothing was issued first. The behaviour the stale read happened to give, ready PIM commands that need no
  precharge (MACs and buffer moves) finishing the open row while a refresh waits for its precharge, is now an explicit
  policy in `schedule_request`.

## What `02-hbm3e-8gbps-preset.patch` changes

Adds timing presets `HBM3_8.0Gbps` and `HBM3_8.0Gbps_NPC` (B200's HBM3E data rate) to `src/dram/impl/HBM3-PIM.cpp`.
In AttAcc's 4.8–6.4 Gbps rows, the row and column timings are the DRAMsim3 `HBM2_8Gb_x128.ini` ns values rounded up
to cycles (tCL/tRCD/tRP 14, tRAS 34, tRC 48, tWR 16, tRTPS 4, tRTPL 6, tCWL 4, tWTRS 6, tWTRL 8, tFAW 30 ns, tREFI
3.9 µs). tRREFD 8 ns and tRFCsb 200 ns equal JESD238B.01 Table 93 (tRREFD, tRFCpb of the 16 Gb die). nCCDS 2 and
nCCDL 4 are the Table 93 minima. nRRDS 2, nRRDL 4 and nRTW 3 are fixed counts from Ramulator 2's HBM3 preset
(`src/dram/impl/HBM3.cpp:31-36` @b7c7027); they do not enter the all-bank PIM rules. The 8.0 Gbps rows use the same ns
values at tCK 0.5 ns.
nCCDL is 5 cycles in both 8.0 Gbps rows: JESD238B.01 Table 93 sets tCCDL >= max(4 nCK, 2.5 ns), which AttAcc's fixed
4 cycles meet up to 6.4 Gbps (2.5 ns) but not at 8.0 Gbps (2.0 ns). nCCDL 5 instead of 4 changes the 8.0 Gbps PIM run times by -0.05 to
+0.24 % over the 27 points of the MAC sweep (`dram/checks/nccdl_check.py`).
The power-limited MAC interval (nCCDAB 9, nCCDSB 10 = 4.5 / 5 ns) is the shortest that is not faster than the 6.4 Gbps
preset (4.375 / 5 ns): 3 % longer, about 5 % less PIM bandwidth and power at 2K. AttAcc gives no value for this rate, so
the choice has no source; the sweep also covers 7 and 6 cycles (mac_sweep_HBM3_8.0Gbps.csv). The `_NPC`
row sets nCCDAB and nCCDSB to the timing limit, 5 cycles (tCCDL; the other `_NPC` rows' 4 cycles would be 2.0 ns here).
The scripts take this limit from `dram/attacc/refresh_sweep.py` (`N_TIMING`: 4 / 4 / 5 cycles at
5.2 / 6.4 / 8.0 Gbps) and does not use intervals below it.

## What `03-concurrent-reads.patch` changes

For running PIM and ordinary reads together (a GPU streaming FC weights while PIM runs attention). All defaults keep the
behaviour of 00-02 (the refresh sweep reproduces cycle for cycle).

- frontend `PIMDualTrace` (`src/frontend/impl/memory_trace/pim_dual_trace.cpp`, new): a PIM trace (`path`) and a read
  stream (`read_path`, one address per line) sent independently, so a full PIM buffer does not hold back the reads;
  `read_count` reads (or, with 0, reads until the PIM trace is sent), `read_rate` reads per cycle (< 0 unlimited)
- controller option `pim_priority` (default true): false checks the read/write buffers before the PIM buffer
- memory-system option `dual_row_buffer` (default false): a second DRAM state for PIM commands, so PIM keeps its own open
  rows (NeuPIMs-style dual row buffers) and shares only the controller's command slots with reads (one row and one
  column command per channel per cycle). Bank-array conflicts between the two are not modelled; use without refresh.

## Check

`dram/attacc/refresh_cmd_check.py` runs the sweep's configurations with `HBM3TraceRecorder` and checks every
issued command against the refresh rules (no activate inside a refresh, no MAC on a closed channel, tRP / tRREFD /
tRFCsb spacing, one refresh per bank per tREFI, no starved refresh), and that the spread schedule is refused. It covers
all-bank refresh and both per-bank modes: `pbj` (JESD238B.01 tRFCpb, tRREFD per pseudo channel) and `pbh`
(HBM2-derived tRFCsb, tRREFD per SID). Output: `dram/checks/out/refresh_cmd_check.txt`.
