"""Spreader-size sensitivity of the bank model (grid 64, k 1, r_convec 0.01361): s_spreader 45.6 mm (repo runs,
runs_recal/runs/a50_par_k1_r0.01361_g64_*) against 70 mm (scratch runs). Unit sources: uni / array 200 W per stack,
gpu 1000 W. Readouts: hottest DRAM grid cell inside the stack footprints (all 8 DRAM dies), and the bottom-die
profile across the left-middle stack (outer edge vs GPU-facing edge)."""
import json, os
import numpy as np
W = os.path.dirname(os.path.abspath(__file__))
REPO = "<analysis>/pipeline/e2e_joint/runs_recal/runs"
WP, HP, n = 0.04553, 0.03657, 64
st = json.load(open(os.path.join(W, "geom_s070", "blocks.json")))["stacks"]
xc = (np.arange(n) + 0.5) * WP / n; yc = HP - (np.arange(n) + 0.5) * HP / n
X, Y = np.meshgrid(xc, yc)
inside = np.zeros((n, n), bool)
for (x0, y0, x1, y1) in st.values():
    inside |= (X > x0) & (X < x1) & (Y > y0) & (Y < y1)
def repo(src):
    return dict(np.load(os.path.join(REPO, f"a50_par_k1_r0.01361_g64_{src}", "fields.npz")))
def mine(src):
    out = {}
    for l in [2] + [4 + 2 * d for d in range(8)]:
        p = os.path.join(W, "runs", src, f"{src}.grid.steady[{l}]")
        out[f"L{l}"] = np.array([float(s.split()[1]) for s in open(p) if s.strip()]).reshape(n, n) - 273.15 - 50.0
    return out
def peak(F):
    return max(F[f"L{4 + 2 * d}"][inside].max() for d in range(8))
def edge_profile(F):
    i = int((HP - 0.0183) / HP * n)
    row = F["L4"][i]; left = xc < 0.008
    return float(row[left][0]), float(row[left][-1])
res = {}
for tag, get in (("s45.6mm", repo), ("s70mm", mine)):
    U, A = get("uni"), get("array")
    try:
        G = get("gpu")
    except (FileNotFoundError, ValueError):
        G = None
    res[tag] = dict(uni_peak_per_W_stack=peak(U) / 200, array_peak_per_W_stack=peak(A) / 200,
                    array_over_uni=peak(A) / peak(U), gpu_peak_dram_per_W=(peak(G) / 1000 if G else None),
                    uni_left_stack_outer_vs_gpu_edge=[v / 200 for v in edge_profile(U)])
print(json.dumps(res, indent=1))
json.dump(res, open(os.path.join(W, "spreader_result.json"), "w"), indent=1)
