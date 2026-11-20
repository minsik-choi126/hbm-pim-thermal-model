from vcommon import *
sol = json.load(open(os.path.join(HERE, "sol.json")))
for nm, (k, r) in sol.items():
    use_pkg(nm)
    HS.run_all(jobs(k, r, ("gpu", "uni", "base", "array", "edge")), workers=5, label=nm)
print("DONE", flush=True)
