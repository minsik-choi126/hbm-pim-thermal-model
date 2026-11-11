"""128 banks per DRAM die (JESD238 8-High 16 Gb: 16 channels x 64 banks over 8 dies) at grid 128, against the code's 64.

Same floorplan rules as thermal_bank_pim.die_blocks (a = 0.50, stripe c = 0.20 parallel to the GPU edge, decoder strips
taking the same fraction of each bank cell), but 4 x 16 banks per half instead of 4 x 8 (each bank half as tall).
Only the PIM-in-bank-arrays map is run (40 W/stack, no base power); the repo's a50_par_g128 base and uniform runs carry
the same physical base and uniform maps, and the model is linear, so the hottest-cell factor is
f = max(T_base + T_array128) - max(T_base) over max(T_uniform) - max(T_base).

    python3 bank128.py run
    python3 bank128.py
"""
import json
import math
import os
import shutil
import subprocess
import sys

import numpy as np

REPO = "<analysis>/pipeline/e2e_joint"
sys.path.insert(0, REPO)
import thermal_bank_pim as B  # noqa: E402
import thermal_pim_sim as TP  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(REPO, "runs_bank", "geom", "a50_par_g128")
NQ = int(os.environ.get("NQ", "16"))
NP = int(os.environ.get("NP", "4"))
HOT = (1, 2)            # stacks that hold the hottest DRAM cell in the repo's array / uniform / base runs (grid 128)
TAG = f"bank{2 * NP * NQ}hot" + ("" if NP == 4 else f"_np{NP}")
G = os.path.join(HERE, "geom", f"{TAG}_a50_par_g128")
NAME = f"a50_par_g128_{TAG}_array"
OUT = os.path.join(HERE, "runs", NAME)
B36 = "0123456789abcdefghijklmnopqrstuvwxyz"


def b36(i):
    return B36[i // 1296] + B36[(i // 36) % 36] + B36[i % 36]


def die_blocks(a, fp):
    c = B.AREA[a]
    out = []
    for s, (x0, y0, x1, y1) in fp.items():
        W, H = x1 - x0, y1 - y0
        if s not in HOT:                                 # HotSpot allows 8192 units: other stacks as 2 x 2 blocks
            for i in range(2):
                for j in range(2):
                    out.append(dict(kind="qq", x=x0 + i * W / 2, y=y0 + j * H / 2, w=W / 2, h=H / 2, stack=s))
            continue
        Lp, Lq = W, H                                    # par
        wc = c * Lp
        bp, bq = (Lp - wc) / 2 / NP, Lq / NQ
        k = 1 - math.sqrt(a / (1 - c))
        wcd, hrd = k * bp, k * bq

        def put(kind, p0, q0, dp, dq):
            out.append(dict(kind=kind, x=x0 + p0, y=y0 + q0, w=dp, h=dq, stack=s))
        for half in (0, 1):
            for iq in range(NQ):
                q0 = iq * bq
                put("rd", 0.0 if half == 0 else Lp / 2 + wc / 2, q0, (Lp - wc) / 2, hrd)
                for ip in range(NP):
                    if half == 0:
                        cell0 = Lp / 2 - wc / 2 - (ip + 1) * bp
                        cd0, arr0 = cell0 + bp - wcd, cell0
                    else:
                        cell0 = Lp / 2 + wc / 2 + ip * bp
                        cd0, arr0 = cell0, cell0 + wcd
                    put("cd", cd0, q0 + hrd, wcd, bq - hrd)
                    put("ar", arr0, q0 + hrd, bp - wcd, bq - hrd)
            put("ts", Lp / 2 - wc / 2 if half == 0 else Lp / 2, 0.0, wc / 2, Lq)      # one stripe block per half
    return out


def build():
    if os.path.isdir(G):
        return json.load(open(os.path.join(G, "blocks.json")))
    tmp = G + ".tmp"
    shutil.rmtree(tmp, ignore_errors=True)
    shutil.copytree(SRC, tmp)
    m0 = B.load(SRC)
    blocks = die_blocks(0.50, m0["stacks"])
    for s, (x0, y0, x1, y1) in m0["stacks"].items():
        ar = sum(b["w"] * b["h"] for b in blocks if b["stack"] == s)
        assert abs(ar / ((x1 - x0) * (y1 - y0)) - 1) < 1e-9
        arr = sum(b["w"] * b["h"] for b in blocks if b["stack"] == s and b["kind"] == "ar")
        assert s not in HOT or abs(arr / ((x1 - x0) * (y1 - y0)) - 0.50) < 1e-9
    old = set(m0["info"])
    meta = []
    n = 0
    for d in range(8):
        p = os.path.join(tmp, f"mem_bank_{d + 1}.flp")
        lines = open(p).read().splitlines()
        keep = [ln for ln in lines if not (ln.split() and ln.split()[0] in old)]
        new = []
        for b in blocks:
            nm = str(d) + b36(n)
            n += 1
            new.append(f"{nm}\t{b['w']:.9f}\t{b['h']:.9f}\t{b['x']:.9f}\t{b['y']:.9f}")
            meta.append(dict(b, name=nm, die=d, layer=4 + 2 * d))
        open(p, "w").write("\n".join([keep[0]] + new + keep[1:]) + "\n")
    m = dict(stacks=m0["stacks"], blocks=meta, nq=NQ, np=NP)
    json.dump(m, open(os.path.join(tmp, "blocks.json"), "w"))
    os.rename(tmp, G)
    return m


def run():
    m = build()
    cols = TP.columns(G)
    v = dict.fromkeys(cols, 0.0)
    for s in range(TP.STACKS):
        tg = [b for b in m["blocks"] if b["stack"] == s and b["kind"] in ("ar", "qq")]
        A = sum(b["w"] * b["h"] for b in tg)
        for b in tg:
            v[b["name"]] += B.PIM_W * b["w"] * b["h"] / A
    os.makedirs(OUT, exist_ok=True)
    pt = os.path.join(OUT, NAME + ".ptrace")
    head = "\t".join(cols)
    vals = "\t".join(("%.4g" % v[c]) if v[c] else "0" for c in cols)
    assert len(head) < 65000 and len(vals) < 65000, (len(head), len(vals))
    open(pt, "w").write(head + "\n" + vals + "\n")
    cmd = [TP.R2.HOTSPOT, "-c", "run.config", "-f", "core_and_mem_ctrl.flp", "-p", pt, "-model_type", "grid",
           "-grid_layer_file", "stack.lcf", "-detailed_3D", "on", "-steady_file", os.path.join(OUT, NAME + ".steady"),
           "-grid_steady_file", os.path.join(OUT, NAME + ".grid.steady")]
    with open(os.path.join(OUT, NAME + ".log"), "w") as lg:
        subprocess.check_call(cmd, cwd=G, stdout=subprocess.DEVNULL, stderr=lg)
    os.remove(pt)


def analyse():
    m = B.load(SRC)
    root = os.path.join(REPO, "runs_bank", "runs")
    GL = {n: B.grid_layers(f"a50_par_g128_{n}", 128, root) for n in ("base", "uniform", "array")}
    A = B.grid_layers(NAME, 128, os.path.join(HERE, "runs"))
    amb = TP.AMB
    comb = {i: GL["base"][i] + (A[i] - amb) for i in GL["base"]}
    rb = B.readouts(GL["base"], m, 128)["cell"]
    u = B.readouts(GL["uniform"], m, 128)["cell"] - rb
    f128 = (B.readouts(comb, m, 128)["cell"] - rb) / u
    f64 = (B.readouts(GL["array"], m, 128)["cell"] - rb) / u
    out = dict(banks_per_die=2 * NP * NQ, f_array_banks=f128, f_array_64_repo=f64, ratio=f128 / f64)
    p = os.path.join(HERE, "bank128_result.json")
    allr = json.load(open(p)) if os.path.exists(p) else {}
    allr[f"np{NP}_nq{NQ}"] = out
    json.dump(allr, open(p, "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    run() if sys.argv[1:] == ["run"] else analyse()
