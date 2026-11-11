"""Family calibrated to the second H100 card's slopes (paths_multi.json 'H100 card2' s_host, s_rail; regression replicas
on the model card's 89 points as in hotspot_recal), then x and beta at the measured row-miss ratio."""
import json, os, sys
REPO = "<analysis>/pipeline/e2e_joint"; sys.path.insert(0, REPO); os.chdir(REPO)
import hotspot_recal as HR
HR.pts()
pm = json.load(open("paths_multi.json"))["H100 card2"]
HR.TARGET["a_G"], HR.TARGET["a_M"] = pm["s_host"], pm["s_rail"]
op = HR.op_beta_decode()
rows = HR.family("cell", op_beta=op)
ok = [r for r in rows if r.get("x") is not None]
for r in rows:
    print({k: (round(r[k], 4) if isinstance(r.get(k), float) else r.get(k)) for k in ("k", "x", "r_convec", "a_G", "a_M", "ratio", "beta_every_bank", "gpu_die_per_nonmem_W")}, flush=True)
res = dict(target=dict(a_G=pm["s_host"], a_M=pm["s_rail"], ratio=pm["ratio"]),
           at_ratio=HR.point_at(ok, HR.RM["main"]["excess"]["ratio"], "ratio"), x0=HR.point_at(ok, 0.0, "x"), x1=HR.point_at(ok, 1.0, "x"))
out = {k: (None if v is None else {kk: v[kk] for kk in ("x", "k", "r_convec", "ratio", "beta_every_bank", "gpu_die_per_nonmem_W")}) if k != "target" else v for k, v in res.items()}
json.dump(out, open("<audit>/validation/e2_card2.json", "w"), indent=1)
print(json.dumps(out, indent=1))
