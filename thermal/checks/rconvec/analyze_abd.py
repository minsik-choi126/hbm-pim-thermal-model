"""A (r_convec sweep), B (can k, x recover the measured slopes at r 0.1 / 0.143) and D (beta direction with r).
Writes out_abd.json."""
import json
import os

import numpy as np

from common import HERE, RC, B, HS, sink_mean_per_W, vset, evaluate, measured, run_name, dump
from run_batch import SRC5, RA, KA, KB, RB

meas = measured()

# validation of the rebuilt binary against the cached run
F0 = HS.fields("a50_par_k0.95369_r0.01102_g64_gpu")
F1 = HS.fields("val_a50_par_k0.95369_r0.01102_g64_gpu")
val = max(float(np.abs(F0[k] - F1[k]).max()) for k in F0)

A = []
for k in KA:
    for r in RA:
        V = vset(k, r, SRC5)
        ev = evaluate(V, 0.0, meas)
        sm = sink_mean_per_W(r)
        ev.update(k=k, r_convec=r, sink_mean_per_W=sm,
                  err_a_G=ev["a_G"] / meas["a_G"] - 1, err_a_M=ev["a_M"] / meas["a_M"] - 1,
                  err_gpu=ev["gpu_per_nonmem_W"] / meas["gpu_G"] - 1,
                  local_pim=ev["pim_rise_per_W"] - 0.0, local_mem=ev["mem_rise_per_W"] - 0.0)
        A.append(ev)

# B: k in 0.3-4 (and the A k values) at r 0.1 / 0.143, x in [0, 1]
xs = np.round(np.linspace(0, 1, 101), 3)
Bres = {}
for r in RB:
    rows = []
    for k in sorted(set(KB) | set(KA)):
        V = vset(k, r, ("gpu", "uni", "base", "array"))
        for x in xs:
            f = RC.model_fit(V, float(x))
            eG, eM = f["a_G"] / meas["a_G"] - 1, f["a_M"] / meas["a_M"] - 1
            rows.append(dict(k=k, x=float(x), a_G=f["a_G"], a_M=f["a_M"], gpu=f["gpu_die_per_nonmem_W"], ratio=f["ratio"],
                             err_a_G=eG, err_a_M=eM, err_max=max(abs(eG), abs(eM)), err_rss=float(np.hypot(eG, eM))))
    best_max = min(rows, key=lambda d: d["err_max"])
    best_rss = min(rows, key=lambda d: d["err_rss"])
    best_aG = min(rows, key=lambda d: d["a_G"])
    best_aM = min(rows, key=lambda d: abs(d["err_a_M"]))
    Bres[f"r{r:g}"] = dict(best_minimax=best_max, best_rss=best_rss, lowest_a_G=best_aG, best_a_M_only=best_aM,
                           a_G_range=[min(d["a_G"] for d in rows), max(d["a_G"] for d in rows)],
                           a_M_range=[min(d["a_M"] for d in rows), max(d["a_M"] for d in rows)],
                           gpu_range=[min(d["gpu"] for d in rows), max(d["gpu"] for d in rows)],
                           n=len(rows), ks=sorted(set(KB) | set(KA)))

# bound: GPU slope >= sink mean = r + t_sink / (k_sink s_sink^2); check on every A run (GPU die max per W of GPU power)
bound = [dict(k=d["k"], r=d["r_convec"], gpu_unit_max=float(vset(d["k"], d["r_convec"], ("gpu",))["gpu"]["gpu"].max()),
              sink_mean=d["sink_mean_per_W"]) for d in A]

out = dict(measured=meas, validation_max_abs_diff_K_per_1000W=val, A=A, B=Bres, bound=bound)
dump(os.path.join(HERE, "out_abd.json"), out)

print(f"validation max |diff| {val:.2e} K (1000 W unit run)")
print("k       r       a_G     a_M     M/G    gpu     ratio  beta   e_x    HBMrise GPUrise sinkmean")
for d in A:
    print(f"{d['k']:<7g} {d['r_convec']:<7g} {d['a_G']:.4f} {d['a_M']:.4f} {d['a_M_over_a_G']:.2f} {d['gpu_per_nonmem_W']:.4f} "
          f"{d['rowmiss_ratio']:.3f} {d['beta']:.3f} {d['e_x'] if d['e_x'] else float('nan'):.3f} {d['top_hbm_rise']:6.1f} "
          f"{d['top_gpu_rise']:6.1f} {d['sink_mean_per_W']:.4f} pim {d['pim_rise_per_W']:.4f} mem {d['mem_rise_per_W']:.4f}")
t = meas["top"]
print(f"measured: a_G {meas['a_G']:.4f} a_M {meas['a_M']:.4f} M/G {meas['a_M'] / meas['a_G']:.2f} gpu {meas['gpu_G']:.4f} ratio {meas['ratio']:.3f}; "
      f"top point {t['board_W']:.0f} W (nonmem {t['nonmem_W']:.0f}, mem {t['mem_W']:.0f}) HBM {t['thbm']} rise {t['hbm_rise_meas']:.1f} "
      f"(fit {t['hbm_rise_fit']:.1f}), GPU {t['tgpu']} rise {t['gpu_rise_meas']:.1f} (fit {t['gpu_rise_fit']:.1f}), c_gpu {meas['c_gpu']:.2f}")
for key, v in Bres.items():
    print(key, "minimax", {q: round(v["best_minimax"][q], 4) for q in ("k", "x", "a_G", "a_M", "err_a_G", "err_a_M", "gpu")},
          "a_G range", [round(z, 4) for z in v["a_G_range"]], "a_M range", [round(z, 4) for z in v["a_M_range"]])
for b in bound:
    print("bound", b)
