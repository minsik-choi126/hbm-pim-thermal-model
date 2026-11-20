from analyze_verify_base import *
CL = "<scratch>/rconvec/clone/thermal/runs/runs"
SUF = {"sp60_sk120": "_sp60_sk120_tsp1", "sp80_sk150": "_sp80_sk150_tsp1", "sp60_sk120_t3": "_sp60_sk120_tsp3"}
KS, RS = (0.9, 1.0), (0.004, 0.006, 0.008)
sol = {}
for nm in SUF:
    use_pkg(nm)
    a = HS.fields(f"v_{G.tag(1.0, 0.010, GRID, FP)}_gpu"); b = dict(np.load(os.path.join(CL, f"{G.tag(1.0, 0.010, GRID, FP)}{SUF[nm]}_gpu", "fields.npz")))
    print(nm, "binary/config cross-check vs earlier cache (k1 r0.010 gpu): max|diff|", max(float(np.abs(a[q]-b[q]).max()) for q in a))
    tab = {(k, r): own(vset(k, r, ("gpu", "uni")) | {"base": vset(k, r, ("uni",))["uni"]}, 0.0)[:2] for k in KS for r in RS}
    for k in KS:
        print("  k", k, " ".join(f"r{r}: aG {tab[k, r][0]:.4f} aM {tab[k, r][1]:.4f}" for r in RS))
    def f(k, r):
        t = (np.log(k) - np.log(KS[0])) / (np.log(KS[1]) - np.log(KS[0]))
        out = []
        for i in range(2):
            vals = [(1 - t) * tab[KS[0], rr][i] + t * tab[KS[1], rr][i] for rr in RS]
            out.append(np.polyval(np.polyfit(RS, vals, 2), r))
        return out
    from scipy.optimize import fsolve
    k, r = fsolve(lambda z: [f(z[0], z[1])[0] / TG["a_G"] - 1, f(z[0], z[1])[1] / TG["a_M"] - 1], [0.95, 0.006])
    print(f"  solution x=0: k {k:.4f} r_convec {r:.5f}")
    sol[nm] = (float(k), float(r))
json.dump(sol, open("sol.json", "w"), indent=1)
