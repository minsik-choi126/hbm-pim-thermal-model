#!/usr/bin/env bash
# DRAM/PIM side of the model.
#   bash dram/run.sh quick   derived values from the committed simulator outputs (+ HBM-Power fetch) into dram/build/quick,
#                            then check them (the committed dram/results stay untouched)
#   bash dram/run.sh full    build AttAcc, rerun every simulation into dram/build/full, compare with dram/results
#   bash dram/run.sh check   check only
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PYTHON:-python3}
step() { local s=$SECONDS; echo "== $*"; "$PY" -m "$@"; echo "   ($((SECONDS - s)) s)"; }

derived() {
  [ -f dram/external/HBM-Power/main_config_HBM3E_6400MTs_power.json ] || bash dram/setup/hbm_power.sh
  step dram.attacc.pim_table
  step dram.hbm_power.idd7_chips
  step dram.hbm_power.hbm3e_split
  step dram.bank_power.stack_power
}

case "${1:-quick}" in
  quick)
    export DRAM_RESULTS=dram/build/quick
    derived
    "$PY" -m dram.check ;;
  full)
    [ -x dram/external/attacc_simulator/ramulator2/ramulator2 ] || bash dram/setup/attacc.sh
    export DRAM_RESULTS=dram/build/full
    for p in HBM3_5.2Gbps HBM3_6.4Gbps HBM3_8.0Gbps; do step dram.attacc.mac_sweep "$p"; done
    step dram.attacc.e2e
    step dram.attacc.refresh_sweep
    step dram.attacc.refresh_cmd_check
    step dram.attacc.concurrent_reads
    derived
    "$PY" -m dram.check ;;
  check)
    "$PY" -m dram.check ;;
  *) echo "usage: bash dram/run.sh [quick|full|check]"; exit 2 ;;
esac
