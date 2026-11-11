# DRAM/PIM simulation (dram/) -> HotSpot package model (thermal/). check, quick and full end with the three checks
# (dram, thermal, parameter tables against code and data). Run times: README.md.
#   make check     DRAM values; HotSpot build, measured fits, one HotSpot run; parameter tables
#   make quick     + thermal calibration, bank model (grid 64), recalibration, beta
#   make full      + AttAcc rebuilt and every DRAM simulation rerun; thermal model on grid 128
#   make figures   redraw figures/ from the HotSpot runs of quick or full
PY      ?= python3
WORKERS ?= 8
export PYTHON := $(PY)
export WORKERS

.PHONY: check quick full figures check-params clean

check:
	bash dram/run.sh quick
	$(PY) -m thermal check
	$(PY) tools/check_params.py

quick:
	bash dram/run.sh quick
	$(PY) -m thermal quick
	$(PY) tools/check_params.py

full:
	bash dram/run.sh full
	$(PY) -m thermal full
	$(PY) tools/check_params.py

check-params:
	$(PY) tools/check_params.py

figures:
	$(PY) -m thermal.figures

# removes simulator checkouts, builds, HotSpot runs and outputs; keeps committed results and figures
clean:
	rm -rf dram/external dram/build thermal/runs thermal/out
	$(MAKE) -C thermal/hotspot clean
