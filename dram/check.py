"""Check the DRAM/PIM values against results/reference.json, and regenerated simulator outputs against the committed ones.

python -m dram.check                              # values in dram/results
DRAM_RESULTS=dram/build/full python -m dram.check # values in a full rerun, and its CSVs against dram/results
Exit status 1 on any mismatch.
"""
import csv
import json
import os
import sys

from dram import common as C

E2E = "attacc_e2e_h100_llama65b.csv"


def get(doc, key):
    for k in key.split("."):
        doc = doc[int(k)] if isinstance(doc, list) else doc[k]
    return doc


def value(docs, file, key):
    if ":" in key:
        op, keys = key.split(":", 1)
        xs = [get(docs[file], k) for k in keys.split(",")]
        return {"min": min, "max": max}[op](xs)
    return get(docs[file], key)


def check_values():
    ref = json.load(open(C.ref("reference.json")))["values"]
    docs = {f: json.load(open(os.path.join(C.RESULTS, f))) for f in sorted({v["file"] for v in ref})
            if os.path.exists(os.path.join(C.RESULTS, f))}
    bad = 0
    for v in ref:
        if v["file"] not in docs:
            print(f"  MISSING {v['file']} in {os.path.relpath(C.RESULTS)} (not produced: run incomplete): {v['name']}")
            bad += 1
            continue
        x = value(docs, v["file"], v["key"])
        if "range" in v:
            ok = v["range"][0] <= x <= v["range"][1]
        elif v.get("exact"):
            ok = x == v["value"]
        else:
            ok = abs(x - v["value"]) <= v["tol"] + 1e-12
        bad += not ok
        shown = f"{x:.4g}" if isinstance(x, float) else str(x)
        print(f"  {'ok  ' if ok else 'FAIL'} {v['name']:58s} {str(v['value']):>6s}  computed {shown:>9s} {v['unit']}")
    return bad, len(ref)


def check_regenerated():
    """CSV outputs of a rerun (DRAM_RESULTS) against the committed ones: byte for byte; the e2e CSV row by row."""
    if os.path.realpath(C.RESULTS) == os.path.realpath(C.REFERENCE):
        return 0, 0
    bad = n = 0
    for f in sorted(os.listdir(C.RESULTS)):
        if not f.endswith(".csv") or not os.path.exists(C.ref(f)):
            continue
        n += 1
        new, old = open(os.path.join(C.RESULTS, f)).read(), open(C.ref(f)).read()
        if f == E2E:
            rows_new = list(csv.reader(new.splitlines()))
            rows_old = {tuple(r[:7]): r for r in csv.reader(old.splitlines())}
            ok = all(rows_old.get(tuple(r[:7])) == r for r in rows_new)
            how = f"{len(rows_new) - 1} rows identical" if ok else "rows differ"
        else:
            ok = new == old
            how = "identical" if ok else "DIFFERENT"
        bad += not ok
        print(f"  {'ok  ' if ok else 'FAIL'} {f:40s} {how}")
    return bad, n


def main():
    print(f"values ({os.path.relpath(C.RESULTS)})")
    b1, n1 = check_values()
    b2, n2 = check_regenerated()
    print(f"{n1 - b1}/{n1} values" + (f", {n2 - b2}/{n2} regenerated files" if n2 else "") + (" OK" if not b1 + b2 else " FAILED"))
    sys.exit(1 if b1 + b2 else 0)


if __name__ == "__main__":
    main()
