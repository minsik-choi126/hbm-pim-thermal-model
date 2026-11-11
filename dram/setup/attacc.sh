#!/usr/bin/env bash
# Build the AttAcc simulator (Ramulator 2 with AttAcc's PIM model) at the pinned commits, with dram/patches/attacc.
#   bash dram/setup/attacc.sh          ->  dram/external/attacc_simulator/ramulator2/ramulator2
# Needs git, cmake >= 3.14 and a C++20 compiler (g++ >= 12 or Apple clang). ATTACC_DIR overrides the location.
# Also makes a venv with numpy and pandas for AttAcc's main.py (dram.attacc.e2e); PYTHON picks the interpreter.
set -euo pipefail
DRAM=$(cd "$(dirname "$0")/.." && pwd)
P=$DRAM/patches/attacc
DEST=${ATTACC_DIR:-$DRAM/external/attacc_simulator}
JOBS=${JOBS:-8}

[ -d "$DEST/.git" ] || git clone -q https://github.com/scale-snu/attacc_simulator.git "$DEST"
cd "$DEST"
git checkout -q "$(cat "$P/AttAcc.commit")"
git submodule update -q --init --recursive
# start from a clean submodule so the script can be rerun
(cd ramulator2 && git reset -q --hard && git clean -qfdx)
# AttAcc's own setup: resets ramulator2 to RAMULATOR2.commit, copies the PIM sources in, applies AttAcc's patches
bash set_pim_ramulator.sh > set_pim_ramulator.log 2>&1
test "$(git -C ramulator2 rev-parse HEAD)" = "$(cat "$P/RAMULATOR2.commit")"
rm -f ramulator.out                 # result cache tracked upstream; main.py rewrites it
cd ramulator2
for p in "$P"/0*.patch; do
  git apply -p2 --whitespace=nowarn "$p"
  echo "applied $(basename "$p")"
done
mkdir -p build && cd build
cmake .. -DCMAKE_POLICY_VERSION_MINIMUM=3.5 > cmake.log 2>&1
make -j"$JOBS" > make.log 2>&1
cp ramulator2 ..
cd "$DEST"
[ -x .venv/bin/python ] || { "${PYTHON:-python3}" -m venv .venv && .venv/bin/pip install -q numpy pandas; }
echo "built $DEST/ramulator2/ramulator2"
