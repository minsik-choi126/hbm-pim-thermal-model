"""Energy balance on the thermal_bank_pim runs (read only): base = GPU 80 W + 5 x 23.9022 W; PIM cases + 5 x 40 W."""
import os, json
import numpy as np
from energy import cfg, heat_out, read_grid, read_inodes
R = "<analysis>/pipeline/e2e_joint/runs_bank"
P_base = 80.0 + 5 * 23.9022
out = {}
for tag, grid in (("a50_par_g64", 64), ("a50_par_g128", 128), ("a75_par_g128", 128), ("a50_perp_g128", 128)):
    c = cfg(os.path.join(R, "geom", tag))
    for case in ("base", "uniform", "array"):
        n = f"{tag}_{case}"
        d = os.path.join(R, "runs", n)
        sink = read_grid(os.path.join(d, f"{n}.grid.steady[21]"), grid)
        ino = read_inodes(os.path.join(d, f"{n}.steady"))
        h = heat_out(sink, ino, c, grid)
        pin = P_base + (0 if case == "base" else 200.0)
        out[n] = dict(P_in=pin, **{k: round(v, 3) for k, v in h.items()}, err_pct=round(100 * (h["total"] / pin - 1), 3))
        print(n, out[n])
json.dump(out, open("eb_bank.json", "w"), indent=1)
