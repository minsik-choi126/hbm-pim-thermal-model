"""D: beta (model/beta.py fixed point) against r_convec at k = 1 for several base-die shares x; out_d.json."""
import os
from common import HERE, B, EVERY, vset, sink_mean_per_W, dump
from run_batch import RA, SRC5
rows = []
for x in (0.0, 0.5, 0.975, 1.0):
    for r in RA:
        V = vset(1.0, r, SRC5)
        fp = B.fixed_point(V, x, EVERY)
        PG, Mh, PP = B.op_at(B.GPU_MAIN, fp["e_x"])
        base = B.temp(V, x, PG, Mh)
        pim = (B.temp(V, x, PG, Mh, PP, EVERY) - base) / PP
        mem = (B.temp(V, x, PG, Mh + PP) - base) / PP
        c = sink_mean_per_W(r)
        rows.append(dict(k=1.0, x=x, r=r, beta=fp["beta"], e_x=fp["e_x"], pim_rise_per_W=pim, mem_rise_per_W=mem, common_sink_mean=c,
                         local_pim=pim - c, local_mem=mem - c))
        print(f"x {x:<5} r {r:<7} beta {fp['beta']:.4f} e_x {fp['e_x']:.3f} pim {pim:.4f} mem {mem:.4f} common {c:.4f} local {pim - c:.4f}/{mem - c:.4f} -> {(pim - c) / (mem - c):.3f}")
dump(os.path.join(HERE, "out_d.json"), rows)
