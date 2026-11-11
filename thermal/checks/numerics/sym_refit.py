"""(c) First-order effect of the symmetric-conductance correction after refitting r_convec: the symmetric build lowers the
model non-memory slope a_G by s (compare_sym.py) at fixed geometry; r_convec is refit so a_G matches 0.0226 again, using
the r dependence of the repo's own unit runs (runs_recal, k fixed, quadratic in r over 0.008-0.020), and the other metrics
are read at the new r (the sym change of each metric added on top). k and x are kept (a second refit of k, x is not done)."""
import json

import numpy as np

from analyze import HR, rnd, vset_repo

SRCS = ("gpu", "uni", "base", "array", "edge")
KEYS = ("a_G", "a_M", "ratio", "beta_every_bank")


def curve(k, x, o):
    rows = []
    for r in HR.RS:
        V = vset_repo(k, r, 64, SRCS)
        f = HR.model_fit(V, x, "cell", False, o)
        rows.append({kk: float(f[kk]) for kk in KEYS})
    return {kk: np.polyfit(HR.RS, [row[kk] for row in rows], 2) for kk in KEYS}, rows


def main():
    o = HR.op_beta_decode()
    sym = json.load(open("out_sym.json"))
    out = {}
    for lab, k, x, r0, symkey in (("x0", 1.0, 0.0, 0.01102, ("x0", "x0")), ("best", 2.0, 0.975, 0.01031, ("best", "x0.975"))):
        c, rows = curve(k, x, o)
        rel = sym[symkey[0]][symkey[1]]["rel"]
        f = lambda kk, r: float(np.polyval(c[kk], r))  # noqa: E731
        target = HR.TARGET["a_G"]
        # solve (1 + rel a_G) a_G_orig(r) = target
        rs = np.linspace(0.004, 0.030, 2601)
        g = np.array([(1 + rel["a_G"]) * f("a_G", r) - target for r in rs])
        i = int(np.argmin(np.abs(g)))
        r1 = float(rs[i])
        r_orig = float(rs[int(np.argmin(np.abs(np.array([f("a_G", r) - target for r in rs]))))])
        rec = dict(k=k, x=x, r_fit_orig_this_k=r_orig, r_fit_sym=r1, sym_rel=rel, rows=rows)
        for kk in KEYS:
            rec[kk] = dict(orig_at_r_orig=f(kk, r_orig), sym_at_r_sym=(1 + rel[kk]) * f(kk, r1),
                           change=(1 + rel[kk]) * f(kk, r1) / f(kk, r_orig) - 1)
        out[lab] = rec
        print(lab, json.dumps(rnd({kk: rec[kk] for kk in KEYS + ("r_fit_orig_this_k", "r_fit_sym")}, 4)))
    json.dump(rnd(out, 6), open("out_sym_refit.json", "w"), indent=1)


if __name__ == "__main__":
    main()
