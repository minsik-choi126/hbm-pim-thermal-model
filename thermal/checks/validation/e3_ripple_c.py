"""Ripple with physical capacitance: HotSpot scales every capacitance by C_FACTOR = 0.333 (temperature.h:114,
RCutil.c:29). With all capacitances x3 every time constant is x3, so the periodic response at on/off times (a, b) equals
the HotSpot-capacitance response at (a/3, b/3). thermal_bank_pim.Ripple on its cached step responses (read only)."""
import json, os, sys
REPO = "<analysis>/pipeline/e2e_joint"; sys.path.insert(0, REPO); os.chdir(REPO)
import thermal_bank_pim as B
ph = json.load(open("thermal_bank_pim.json"))["ripple"]["phases"]
on, off = ph["on"], ph["off"]
t, ton = ph["step_ms"], ph["pim_on_ms"]
out = {}
for spec in B.RIPS:
    R = B.Ripple(spec)
    rec = {}
    for e in ("uniform", B.E_TABLE, 0.30):
        for lab, a, b in (("per_layer", ton / 80, (t - ton) / 80), ("lump", ton, t - ton)):
            r1 = R.case(a, b, on, off, e)
            r3 = R.case(a / 3, b / 3, on, off, e)
            rec[f"{lab}_e{e if isinstance(e, str) else round(e, 4)}"] = dict(hotspot_C=r1["max_above_step_avg"], physical_C=r3["max_above_step_avg"],
                                                                         p2p_hotspot_C=r1["peak_to_peak"], p2p_physical_C=r3["peak_to_peak"])
    out[B.tag(*spec)] = rec
json.dump(out, open("<audit>/validation/e3_ripple_c.json", "w"), indent=1)
print(json.dumps(out, indent=1))
