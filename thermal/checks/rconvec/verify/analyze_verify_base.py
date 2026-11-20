import csv, re
from vcommon import *
S5 = ("gpu", "uni", "base", "array", "edge")
EVERY = RC.PIM_MAPS["every_bank"]
meas = json.load(open(os.path.join(HERE, "out", "measured.json")))["h100_fit"]
TG = dict(a_G=meas["s_host"], a_M=meas["s_rail"], gpu=meas["gpu_s_host"])
# own measured rows
D = os.path.join(REPO, "thermal/data/measured/h100_exp1"); pat = re.compile(r"^h100_(anchor_r1|anchor_r2|anchor_r3desc|combo2d[a-z0-9_]*|memsweep[a-z0-9_]*)\.csv$")
rows = []
for f in sorted(os.listdir(D)):
    if pat.match(f) and "samples" not in f:
        for r in csv.DictReader(open(os.path.join(D, f))):
            if r.get("settled", "1") == "1" and r.get("mem_power_w") and r["mode"] != "idle":
                p, m = float(r["power"]), float(r["mem_power_w"]); rows.append((p - m, m, p))
H = np.array([r[0] for r in rows]); Mw = np.array([r[1] for r in rows])
X = np.column_stack([np.ones(len(H)), H, Mw])
top = max(rows, key=lambda r: r[2])

def own(V, x):
    cell = H[:, None] * V["gpu"]["cell"] + (Mw[:, None] / 5) * ((1 - x) * V["uni"]["cell"] + x * V["base"]["cell"])
    gpu = H[:, None] * V["gpu"]["gpu"] + (Mw[:, None] / 5) * ((1 - x) * V["uni"]["gpu"] + x * V["base"]["gpu"])
    b = np.linalg.lstsq(X, cell.max(1), rcond=None)[0]; g = np.linalg.lstsq(X, gpu.max(1), rcond=None)[0]
    t = (top[0] * V["gpu"]["cell"] + top[1] / 5 * ((1 - x) * V["uni"]["cell"] + x * V["base"]["cell"])).max()
    return b[1], b[2], g[1], t

def report(lab, V, x=0.0, beta=True):
    f = RC.model_fit(V, x); aG, aM, gG, t = own(V, x)
    s = f"{lab}: repo model_fit a_G {f['a_G']:.5f} a_M {f['a_M']:.5f} gpu {f['gpu_die_per_nonmem_W']:.5f} | own a_G {aG:.5f} a_M {aM:.5f} gpu {gG:.5f} | top HBM {meas['c']+t:.1f} C (rise {t:.1f})"
    s += f" | err a_G {f['a_G']/TG['a_G']-1:+.3%} a_M {f['a_M']/TG['a_M']-1:+.3%} gpu {f['gpu_die_per_nonmem_W']/TG['gpu']-1:+.1%}"
    if beta:
        fp = B.fixed_point(V, x, EVERY); s += f" | beta {fp['beta']:.4f} e_x(fp) {fp['e_x']:.4f} threshold(beta) {B.threshold(B.GPU_MAIN, fp['beta']):.4f}"
    print(s, flush=True)

