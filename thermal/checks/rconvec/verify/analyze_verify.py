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

use_pkg("default")
CL = "<scratch>/rconvec/clone/thermal/runs/runs"
for k, r in ((0.95369, 0.01102), (0.95369, 0.143)):
    Vv = vset(k, r, S5); report(f"default k{k} r{r}", Vv)
    for s in ("gpu", "uni"):
        a = HS.fields(f"v_{G.tag(k, r, GRID, FP)}_{s}"); bpath = os.path.join(CL, f"{G.tag(k, r, GRID, FP)}_{s}", "fields.npz")
        if os.path.exists(bpath):
            b = dict(np.load(bpath)); print(f"   fresh vs earlier cache {s}: max|diff| {max(float(np.abs(a[q]-b[q]).max()) for q in a):.2e} K")
for k, r in ((0.3, 0.1), (4.0, 0.143)):
    Vv = vset(k, r, ("gpu", "uni", "base"))
    best = min(((RC.model_fit(Vv, x)["a_G"], x) for x in np.linspace(0, 1, 101)))
    print(f"k{k} r{r}: min a_G over x in [0,1] = {best[0]:.4f} at x {best[1]:.2f} ({best[0]/TG['a_G']:.2f}x measured)")
PL = {"sp60_sk120": (0.9463129047243255, 0.00531352990229107), "sp80_sk150": (0.9474266215923582, 0.0023038324629938807),
      "sp60_sk120_t3": (0.9556582763747883, 0.0053292807503861435)}
for nm, (k, r) in PL.items():
    use_pkg(nm)
    cfg = open(os.path.join(paths.RUNS, "geom", G.tag(k, r, GRID, FP), "run.config")).read()
    vals = {q: re.search(r"-%s\s+(\S+)" % q, cfg).group(1) for q in ("s_spreader", "s_sink", "t_spreader", "r_convec")}
    print(nm, vals)
    report(f"{nm} k{k:.4f} r{r:.5f}", vset(k, r, S5))
