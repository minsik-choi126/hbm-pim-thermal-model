"""python -m thermal {build|check|quick|full} [--all-k]

  build   compile HotSpot (thermal/hotspot, SUPERLU off)
  check   build + measured fits + one HotSpot run against the stored reference
  quick   + calibration (k = 1), bank model and ripple on grid 64 (30 ms windows),
          recalibration for k = 0.75 - 2.0 without the channel subsets, die-average readouts,
          beta at x = 0 (direct runs, grid 64) and along the recalibration family
  full    + grid 128, 100 ms ripple window, recalibration over all k with the subsets, beta also on grid 128;
          --all-k also calibrates k = 1.5 ... 16
Run times: thermal/README.md. Every mode ends with python -m thermal.check.
Environment: WORKERS, HOTSPOT, THERMAL_RUNS, THERMAL_OUT.
"""
import json
import os
import platform
import subprocess
import sys
import time

from . import paths


def build():
    if os.path.exists(paths.HOTSPOT) and os.environ.get("HOTSPOT"):
        return
    exe = os.path.join(paths.HOTSPOT_SRC, "hotspot")
    srcs = [os.path.join(paths.HOTSPOT_SRC, f) for f in os.listdir(paths.HOTSPOT_SRC) if f.endswith((".c", ".h"))]
    if os.path.exists(exe) and os.path.getmtime(exe) >= max(os.path.getmtime(s) for s in srcs):
        return
    t0 = time.time()
    # -fcommon: GCC 10+ defaults to -fno-common, which breaks the link of HotSpot 6.0's shared globals
    subprocess.check_call(["make", "-C", paths.HOTSPOT_SRC, "hotspot", "SUPERLU=0", "EXTRAFLAGS=-fcommon"],
                          stdout=subprocess.DEVNULL)
    print(f"[build] HotSpot built in {time.time() - t0:.0f} s ({platform.system()} {platform.machine()})", flush=True)


def smoke():
    """One HotSpot run (2 x 2 floorplan, k = 1, r 0.015, GPU 80 W + DRAM map) against the stored block temperatures."""
    from .model import hotspot as HS
    from .model import measured as M
    from .steps import calibrate as C
    an = M.anchor_slopes()
    dm = json.load(open(paths.inputs("dram_map_decode.json")))["per_block_W"]
    job = C.jobs(1.0, 0.015, an["mem_per_board_W"], dm)[0]
    t0 = time.time()
    HS.run_all([job], label="smoke")
    r = C.read(job[0])
    out = dict(run=job[0], dram_peak_C=r["dram_peak"], dram_peak_block=r["dram_peak_block"], dram_mean_C=r["dram_mean"],
               gpu_die_C=r["gpu_die"], seconds=round(time.time() - t0, 1))
    json.dump(out, open(paths.out("smoke.json"), "w"), indent=1)
    print(f"[smoke] {job[0]}: hottest DRAM block {r['dram_peak']:.2f} C ({r['dram_peak_block']}), GPU die {r['gpu_die']:.2f} C", flush=True)


def main(argv):
    mode = argv[0] if argv else "check"
    if mode not in ("build", "check", "quick", "full"):
        print(__doc__)
        return 2
    t0 = time.time()
    build()
    if mode == "build":
        return 0
    from .steps import bank, beta, calibrate, measured, recal
    times = {}

    def step(nm, fn, *a, **kw):
        t = time.time()
        fn(*a, **kw)
        times[nm] = round(time.time() - t, 1)
    step("measured", measured.main)
    if mode == "check":
        step("smoke", smoke)
    else:
        step("calibrate", calibrate.main, calibrate.KS_ALL if "--all-k" in argv else (1.0,))
        if mode == "quick":
            step("bank", bank.main, (64,), {"a50_par": 300, "a75_par": 300}, False)
            step("recal", recal.main, recal.KS_QUICK, False)
            step("beta", beta.main, (64,))
        else:
            step("bank", bank.main, (64, 128), {"a50_par": 1000, "a75_par": 300}, True)
            step("recal", recal.main, recal.KS, True)
            step("beta", beta.main, (64, 128))
    times["total"] = round(time.time() - t0, 1)
    json.dump(dict(mode=mode, workers=paths.WORKERS, seconds=times, platform=f"{platform.system()} {platform.machine()}",
                   utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())), open(paths.out("mode.json"), "w"), indent=1)
    print(f"[{mode}] done in {times['total'] / 60:.1f} min: {times}", flush=True)
    from . import check
    return check.main([])


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
