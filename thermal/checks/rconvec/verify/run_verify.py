import sys
from vcommon import *
S5 = ("gpu", "uni", "base", "array", "edge")
PLAN = {"default": [(0.95369, 0.143, S5), (0.95369, 0.01102, S5), (0.3, 0.1, ("gpu", "uni", "base")), (4.0, 0.143, ("gpu", "uni", "base"))],
        "sp60_sk120": [(0.9463129047243255, 0.00531352990229107, S5)],
        "sp80_sk150": [(0.9474266215923582, 0.0023038324629938807, S5)],
        "sp60_sk120_t3": [(0.9556582763747883, 0.0053292807503861435, S5)]}
for name in sys.argv[1:]:
    use_pkg(name)
    J = []
    for k, r, srcs in PLAN[name]:
        J += jobs(k, r, srcs)
    print(name, paths.RUNS, paths.CONFIG, len(J), flush=True)
    HS.run_all(J, workers=int(os.environ.get("W", "5")), label=name)
print("DONE", flush=True)
