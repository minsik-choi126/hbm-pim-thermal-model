"""Compare the results in out/ with thermal/results/reference.json.

    python -m thermal.check            # mode from out/mode.json (check, quick or full)
    python -m thermal.check --mode full

Each reference value carries the paper's printed value, the expected value per mode and a tolerance. Values whose
step has not run are skipped. Also checks that the inputs taken from dram/results still match it. Exit status 1 if a
value is off or nothing could be checked.
"""
import json
import os
import re
import sys

from . import paths


def get(path):
    """'bank.summary.half[0]' -> value from out/bank.json (None if missing)."""
    head, _, rest = path.partition(".")
    p = paths.out(head + ".json")
    if not os.path.exists(p):
        return None
    v = json.load(open(p))
    for part in re.findall(r"[^.\[\]]+|\[\d+\]", rest):
        if part.startswith("["):
            i = int(part[1:-1])
            v = v[i] if isinstance(v, list) and i < len(v) else None
        else:
            v = v.get(part) if isinstance(v, dict) else None
        if v is None:
            return None
    return v


def mode_of():
    try:
        return json.load(open(paths.out("mode.json")))["mode"]
    except (OSError, KeyError, ValueError):
        return "full"


def handoff():
    """data/inputs taken from dram/results: the bank power map (exact) and the HBM3E read split (6 digits)."""
    dram = os.path.join(os.path.dirname(paths.ROOT), "dram", "results")
    if not os.path.isdir(dram):
        return 0
    bad = 0
    a = json.load(open(paths.inputs("dram_map_decode.json")))["per_block_W"]
    b = json.load(open(os.path.join(dram, "bank_power_stack.json")))["per_block_W"]
    s = json.load(open(paths.inputs("hbm3e_read_split.json")))["patterns"]
    t = json.load(open(os.path.join(dram, "hbm3e_split.json")))["patterns"]
    for name, ok in (("dram_map_decode.json = dram/results/bank_power_stack.json", a == b),
                     ("hbm3e_read_split.json = dram/results/hbm3e_split.json",
                      all(v == float(f"{t[p][k]:.6g}") for p in s for k, v in s[p].items()))):
        print(f"[check] input {name}: {'ok' if ok else 'OFF'}")
        bad += not ok
    return bad


def main(argv):
    mode = argv[argv.index("--mode") + 1] if "--mode" in argv else mode_of()
    ref = json.load(open(os.path.join(paths.RESULTS, "reference.json")))
    rows, bad, n = [], 0, 0
    for it in ref["values"]:
        if mode == "check" and it["id"].split(".")[0] not in ("measured", "smoke"):
            continue                    # check mode makes only these; out/ may hold older quick or full results
        exp = it.get(mode, it.get("full")) if mode != "check" else it.get("full")
        if exp is None:
            continue
        got = get(it["id"])
        if got is None:
            continue
        tol = it["tol"].get(mode, it["tol"].get("full")) if isinstance(it["tol"], dict) else it["tol"]
        ok = abs(got - exp) <= tol
        n += 1
        bad += not ok
        rows.append((it["what"], it.get("paper", ""), exp, got, tol, ok))
    w = max([len(r[0]) for r in rows] + [10])
    print(f"\n{'quantity':{w}s}  {'paper':>8s}  {'expected':>11s}  {'got':>11s}  {'tol':>7s}")
    for what, paper, exp, got, tol, ok in rows:
        print(f"{what:{w}s}  {paper:>8s}  {exp:11.5g}  {got:11.5g}  {tol:7.1g}  {'ok' if ok else 'OFF'}")
    print(f"\n[check] mode {mode}: {n - bad}/{n} values within tolerance", flush=True)
    bad += handoff()
    return 1 if bad or not n else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
