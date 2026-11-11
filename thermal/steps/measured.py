"""Step 1: fits of the measured H100 data (no HotSpot) -> out/measured.json."""
import json
import time

from .. import paths
from ..model import measured as M


def main():
    t0 = time.time()
    rm = M.rowmiss()
    out = dict(h100_fit=M.h100_fit(), anchor=M.anchor_slopes(),
               rowmiss=dict(n=rm["n"], ratio=rm["main"]["ratio"], ratio_ci95=[rm["boot_ratio"]["lo"], rm["boot_ratio"]["hi"]],
                            a=rm["main"]["a"], b=rm["main"]["b"], b_x=rm["main"]["b_x"], m0_W=rm["main"]["m0"],
                            excess_mean_W=rm["main"]["excess_mean_w"], boot_n=rm["boot_ratio"]["n"], seed=rm["seed"]))
    out["seconds"] = round(time.time() - t0, 1)
    json.dump(out, open(paths.out("measured.json"), "w"), indent=1)
    h, r = out["h100_fit"], out["rowmiss"]
    print(f"[measured] H100 fit n {h['n']}: s_host {h['s_host']:.5f} {[round(v, 5) for v in h['s_host_ci']]}, "
          f"s_rail {h['s_rail']:.5f} {[round(v, 5) for v in h['s_rail_ci']]}; row-miss ratio {r['ratio']:.5f} "
          f"{[round(v, 4) for v in r['ratio_ci95']]} ({out['seconds']} s)", flush=True)
    return out


if __name__ == "__main__":
    main()
