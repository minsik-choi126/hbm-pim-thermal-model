import sys
from vcommon import *
for name in ("sp60_sk120", "sp80_sk150", "sp60_sk120_t3"):
    use_pkg(name)
    J = jobs(1.0, 0.010, ("gpu",))   # binary cross-check against the earlier agent's cache
    for k in (0.9, 1.0):
        for r in (0.004, 0.006, 0.008):
            J += jobs(k, r, ("gpu", "uni"))
    HS.run_all(J, workers=5, label=name)
print("DONE", flush=True)
