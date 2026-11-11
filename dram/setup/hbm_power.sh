#!/usr/bin/env bash
# Fetch the HBM-Power inputs (Olgun et al., artifact v0.3-alpha, Zenodo 10.5281/zenodo.21970747) that the
# dram/hbm_power scripts read. Sparse checkout (about 31 MB); the upstream repository has no top-level LICENSE, so
# nothing of it is copied into this repository.
#   bash dram/setup/hbm_power.sh            ->  dram/external/HBM-Power
#   bash dram/setup/hbm_power.sh --runner   also builds the patched DRAMPower HBM3_runner (optional cross-check)
# HBM_POWER_DIR overrides the location.
set -euo pipefail
DRAM=$(cd "$(dirname "$0")/.." && pwd)
P=$DRAM/patches/hbm_power
DEST=${HBM_POWER_DIR:-$DRAM/external/HBM-Power}
URL=https://github.com/CMU-SAFARI/HBM-Power.git
MAIN=61aada9a54ec349efde024b77f868b1a7ebeba0e        # main branch: config/HBM3E_6400MTs/power.json
MAIN_SHA256=8507bfaf4ef6bd48c1ee439550d58429c0fae7aba5ee272ad1fb0bdbe5ecd11c
PATHS=(/data/all_idd_measurements.csv /data/HBM3E_measurements.csv /sources/figure16/configs/
       /sources/figure16/traces/hbm3_random_read_4rpa.csv /sources/figure16/traces/hbm3_baseline_nop_ref.csv
       /scripts/figures/figure16.py /sources/drampower/src/DRAMPower/DRAMPower/util/datapattern_model.h)

[ -d "$DEST/.git" ] || git clone -q --filter=blob:none --no-checkout "$URL" "$DEST"
cd "$DEST"
if [ "${1:-}" = "--runner" ]; then
  git sparse-checkout set --no-cone "${PATHS[@]}" /sources/drampower/
else
  git sparse-checkout set --no-cone "${PATHS[@]}"
fi
git checkout -q "$(head -1 "$P/HBM-Power.commit")"         # tag v0.3-alpha

CFG=main_config_HBM3E_6400MTs_power.json
[ -f "$CFG" ] || curl -fsSL "https://raw.githubusercontent.com/CMU-SAFARI/HBM-Power/$MAIN/config/HBM3E_6400MTs/power.json" -o "$CFG"
sha() { (command -v sha256sum >/dev/null && sha256sum "$1" || shasum -a 256 "$1") | cut -d' ' -f1; }
check() { [ "$(sha "$1")" = "$2" ] || { echo "sha256 mismatch for $1"; exit 1; }; }
check "$CFG" "$MAIN_SHA256"
check data/HBM3E_measurements.csv eddcd6bd5277a7870d794a7377107a7f89a11f7dce83a8da40c4933e4b822607    # H200 read energy
check data/all_idd_measurements.csv 8beb9a4114ed88536ee03f0839696246a632d4dd27889fc0b08e6f9572797e38  # HBM2 IDD loops
echo "HBM-Power $(git rev-parse --short HEAD) + main $CFG in $DEST"

if [ "${1:-}" = "--runner" ]; then
  # 02 carries HBM3_runner.cpp from upstream to its final state (01's change included), so 01 skips that file
  if git apply --reverse --check "$P/02-hbm3-refresh.patch" 2>/dev/null; then echo "patches already applied"; else
    git apply --exclude=sources/drampower/src/cli/main/HBM3_runner.cpp "$P/01-json-per-bank-and-shared.patch"
    git apply "$P/02-hbm3-refresh.patch"
  fi
  cmake -S sources/drampower -B sources/drampower/build -DCMAKE_BUILD_TYPE=Release -DCMAKE_POLICY_VERSION_MINIMUM=3.5 > runner_cmake.log 2>&1
  cmake --build sources/drampower/build -j"${JOBS:-8}" > runner_build.log 2>&1
  ls sources/drampower/build/bin/HBM3_runner
fi
