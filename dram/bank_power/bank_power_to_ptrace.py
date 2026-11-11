#!/usr/bin/env python3
"""DRAMPower per-bank energy  ->  HotSpot ptrace (bank-resolved HBM power map).

The interface between the DRAM side and the thermal side: it turns the per-bank energy of the patched DRAMPower
HBM3_runner (--json) into the power of the floorplan's DRAM blocks, and it is the only place where the placement
assumptions (which pseudo channel sits on which die and block) live.

INPUT  --drampower-json : HBM3_runner --json output (one entry per simulated pseudo channel: banks[{bank, E_*}],
                          shared{...}, elapsed cycles, tCK); several files may be given. dram/bank_power/data holds
                          the two pseudo channels of an H100 Llama-3.1-8B decode trace.
       --geom           : a HotSpot geometry directory with stack.lcf and *.flp (blocks B_*, C_0, LC_*)
OUTPUT <out>.ptrace     : one-row ptrace whose columns are the powered layers' blocks in LCF order (lcf_columns.py)
       <out>.map.json   : the accounting (per-block, per-die, stack and package W, every assumption)

PLACEMENT MODEL (assumptions, recorded in map.json)
  * 8-high stack, dies 0 (bottom, mem_bank_1, next to the base die) .. 7 (top, mem_bank_8).
  * channels_per_stack / 8 channels per die; channel c on die c // ch_per_die.
    Ramulator 2's HBM3 organisation models each pseudo channel as 2 ranks (stack IDs); the trace converter folds the
    rank into the 16-bank index (total energy changes by 0.06 %) and this bridge ignores the SID ('sid_folded').
  * within a die the (channel, pseudo channel, bank group, bank) tuples are enumerated in that order and split evenly
    over the die's floorplan blocks (HBM3, 4 blocks per die: one pseudo channel per block).
  * --replicate uniform: the simulated pseudo channels stand for every pseudo channel of the stack.
  * shared (non-bank) energy is spread over a pseudo channel's banks uniformly or in proportion to bank energy.
  * --stacks N: package = N identical stacks; --geom-stacks: stacks present in the floorplan (B_ index stack-major).
  * PHY / controller power is not in DRAMPower's output; --phy-w spreads it over the base die's LC_* blocks.
  * PIM power: --pim-w per stack, uniformly over B_* (--pim-target uniform), on P_* strips (strip) or on a block list.

Usage
  python3 dram/bank_power/bank_power_to_ptrace.py --drampower-json dram/bank_power/data/*.json \
      --geom <geometry dir> --hbm hbm3 --stacks 5 --geom-stacks 5 --gpu-w 300 --out out/case
stdlib only.
"""
import argparse
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from lcf_columns import build_columns  # noqa: E402

BANK_RE = re.compile(r"^B_(\d+)$")
STRIP_RE = re.compile(r"^P_(\d+)$")
LC_RE = re.compile(r"^LC_\d+(_\d+)?$")

ORG = {
    # channels per stack, pseudo-channels per channel, bank groups per PC, banks per BG
    "hbm3": dict(ch=16, pc=2, bg=4, bank=4),
    "hbm3_sid": dict(ch=16, pc=2, bg=8, bank=4),   # rank/SID kept as bank-group bits (convert --rank-as-bg): 32 banks/PC, 1024/stack
    "hbm2": dict(ch=8, pc=2, bg=4, bank=4),
}
DIES = 8
BLOCKS_PER_DIE = 4          # default (generic 2x2 geometry); overridden from the geometry's B_* count


def load_pc_entries(paths):
    """Flatten all --drampower-json files into a list of pseudo-channel records."""
    out = []
    for p in paths:
        with open(p) as fh:
            j = json.load(fh)
        pcs = j.get("pseudochannels") or j.get("pcs") or []
        if not pcs:
            sys.exit("%s: no 'pseudochannels' array (is this the patched runner's --json?)" % p)
        for rec in pcs:
            rec = dict(rec)
            rec["_src"] = os.path.basename(p)
            out.append(rec)
    return out


def bank_total_pJ(bank):
    """Per-bank energy: the runner's own 'total' (upstream energy_info_t::total(), which excludes
    E_pre_RDA/E_pre_WRA) when present; else the sum of every E_* field."""
    if isinstance(bank.get("total"), (int, float)):
        return float(bank["total"])
    tot = 0.0
    for k, v in bank.items():
        if k.startswith("E_") and isinstance(v, (int, float)):
            tot += float(v)
    return tot


def pc_bank_watts(rec, shared_policy):
    """-> (list of per-bank W in bank order, shared W, elapsed s). Shared energy folded per policy."""
    banks = sorted(rec["banks"], key=lambda b: int(b["bank"]))
    cycles = float(rec["elapsed_cycles"])
    tck_ps = float(rec["tCK_ps"])
    if cycles <= 0 or tck_ps <= 0:
        sys.exit("bad elapsed_cycles/tCK_ps in %s" % rec.get("_src"))
    dur_s = cycles * tck_ps * 1e-12
    e_bank = [bank_total_pJ(b) for b in banks]
    e_shared = float(rec.get("shared_sum_pJ", 0.0))
    if "shared" in rec and not rec.get("shared_sum_pJ"):
        e_shared = sum(float(v) for v in rec["shared"].values() if isinstance(v, (int, float)))
    if shared_policy == "uniform" or sum(e_bank) <= 0:
        add = [e_shared / len(e_bank)] * len(e_bank)
    else:
        s = sum(e_bank)
        add = [e_shared * e / s for e in e_bank]
    w = [(e + a) * 1e-12 / dur_s for e, a in zip(e_bank, add)]
    return w, e_shared * 1e-12 / dur_s, dur_s


def block_of(org, ch, pc, bg, bank, blocks_per_die=BLOCKS_PER_DIE):
    """(channel, pc, bg, bank) -> (die, block-in-die) under the placement model in the docstring."""
    ch_per_die = org["ch"] // DIES
    if ch_per_die < 1:
        sys.exit("organization has fewer channels than dies; placement model does not apply")
    die = ch // ch_per_die
    ch_local = ch % ch_per_die
    per_pc = org["bg"] * org["bank"]
    per_die = ch_per_die * org["pc"] * per_pc
    lin = ((ch_local * org["pc"] + pc) * org["bg"] + bg) * org["bank"] + bank
    blk = lin * blocks_per_die // per_die
    return die, blk


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--drampower-json", nargs="+", required=True)
    ap.add_argument("--geom", required=True)
    ap.add_argument("--hbm", choices=sorted(ORG), default="hbm3")
    ap.add_argument("--stacks", type=int, default=5, help="stacks per package (H100 80GB = 5)")
    ap.add_argument("--geom-stacks", type=int, default=1, help="stacks present in the floorplan (B_ index stack-major); the per-stack map is replicated to each")
    ap.add_argument("--pim-stacks", default="all", help="which geometry stacks get PIM power: all | comma list of indices")
    ap.add_argument("--replicate", choices=["uniform", "none"], default="uniform",
                    help="uniform: simulated PC(s) stand for all PCs of the stack; none: only the "
                         "simulated (channel,pc) get power (full-coverage traces)")
    ap.add_argument("--shared-policy", choices=["uniform", "proportional"], default="proportional")
    ap.add_argument("--gpu-w", type=float, default=80.0, help="power on C_0 (GPU die), W")
    ap.add_argument("--phy-w", type=float, default=0.0, help="host-side PHY/controller W on LC_* (per stack)")
    ap.add_argument("--scale-dram-to", type=float, default=None,
                    help="rescale the DRAM map so the per-stack DRAM total equals this many W")
    ap.add_argument("--pim-w", type=float, default=0.0, help="PIM compute W per stack to inject")
    ap.add_argument("--pim-target", default="uniform",
                    help="uniform | strip | comma list of block names")
    ap.add_argument("--out", required=True, help="output prefix (writes .ptrace and .map.json)")
    args = ap.parse_args()

    org = ORG[args.hbm]
    lcf = os.path.join(args.geom, "stack.lcf")
    if not os.path.isfile(lcf):
        sys.exit("stack.lcf missing in %s" % args.geom)
    cols = build_columns(lcf)
    bank_cols = [c for c in cols if BANK_RE.match(c)]
    strip_cols = [c for c in cols if STRIP_RE.match(c)]
    lc_cols = [c for c in cols if LC_RE.match(c)]
    if len(bank_cols) % DIES:
        sys.exit("bank block count %d is not a multiple of %d dies" % (len(bank_cols), DIES))
    n_geom_stacks = args.geom_stacks                 # stacks present in the floorplan (B_ index is stack-major)
    if len(bank_cols) % (DIES * n_geom_stacks):
        sys.exit("bank block count %d is not a multiple of %d dies x %d stacks" % (len(bank_cols), DIES, n_geom_stacks))
    bpd = len(bank_cols) // DIES // n_geom_stacks    # blocks per die: 4 (generic) or 128 (16x8 PIM grid)
    if "C_0" not in cols:
        sys.exit("no C_0 block in powered layers")

    recs = load_pc_entries(args.drampower_json)
    # per-PC bank watts, keyed by (channel, pc)
    sim = {}
    meta_pc = []
    for rec in recs:
        w, w_shared, dur = pc_bank_watts(rec, args.shared_policy)
        key = (int(rec.get("channel", 0)), int(rec.get("pseudochannel", 0)))
        sim[key] = w
        meta_pc.append(dict(src=rec["_src"], channel=key[0], pseudochannel=key[1],
                            pc_W=sum(w), shared_W=w_shared, duration_s=dur, n_banks=len(w)))
    n_pc_stack = org["ch"] * org["pc"]
    per_pc_banks = org["bg"] * org["bank"]
    for k, w in sim.items():
        if len(w) != per_pc_banks:
            sys.exit("PC %s has %d banks, organization says %d" % (k, len(w), per_pc_banks))

    # assemble the stack map (32 blocks)
    block_w = {c: 0.0 for c in bank_cols}
    mean_pc = [sum(v[i] for v in sim.values()) / len(sim) for i in range(per_pc_banks)]
    covered = 0
    for ch in range(org["ch"]):
        for pc in range(org["pc"]):
            if (ch, pc) in sim:
                w = sim[(ch, pc)]
            elif args.replicate == "uniform":
                w = mean_pc
            else:
                continue
            covered += 1
            for bg in range(org["bg"]):
                for bank in range(org["bank"]):
                    die, blk = block_of(org, ch, pc, bg, bank, bpd)
                    for gs in range(n_geom_stacks):
                        name = "B_%d" % (gs * DIES * bpd + die * bpd + blk)
                        block_w[name] += w[bg * org["bank"] + bank]
    dram_stack_w = sum(block_w.values()) / n_geom_stacks
    scale = 1.0
    if args.scale_dram_to is not None and dram_stack_w > 0:
        scale = args.scale_dram_to / dram_stack_w
        for k in block_w:
            block_w[k] *= scale
        dram_stack_w = sum(block_w.values())

    # PIM injection
    pim = {c: 0.0 for c in cols}
    if args.pim_w > 0:
        pim_stacks = list(range(n_geom_stacks)) if args.pim_stacks == "all" else [int(x) for x in args.pim_stacks.split(",")]
        per_stack = DIES * bpd
        in_pim_stack = lambda c: (int(c.split("_")[1]) // per_stack) in pim_stacks if BANK_RE.match(c) else True
        if args.pim_target == "uniform":
            tgt = [c for c in bank_cols if in_pim_stack(c)]
        elif args.pim_target == "strip":
            if not strip_cols:
                sys.exit("--pim-target strip but geometry has no P_* blocks")
            tgt = strip_cols
        else:
            tgt = [t.strip() for t in args.pim_target.split(",") if t.strip()]
            bad = [t for t in tgt if t not in cols]
            if bad:
                sys.exit("unknown PIM target blocks: %s" % bad)
        n_pim_stacks = len(pim_stacks) if args.pim_target != "strip" else n_geom_stacks
        for t in tgt:
            pim[t] += args.pim_w * n_pim_stacks / len(tgt)      # --pim-w is PER STACK

    phy = {c: (args.phy_w * n_geom_stacks / len(lc_cols) if lc_cols else 0.0) for c in lc_cols}   # --phy-w is PER STACK
    if args.phy_w > 0 and not lc_cols:
        sys.exit("--phy-w given but no LC_* blocks in geometry")

    row = []
    for c in cols:
        v = 0.0
        if c == "C_0":
            v = args.gpu_w
        v += block_w.get(c, 0.0) + pim.get(c, 0.0) + phy.get(c, 0.0)
        row.append(v)

    out_pt = args.out + ".ptrace"
    os.makedirs(os.path.dirname(os.path.abspath(out_pt)), exist_ok=True)
    with open(out_pt, "w") as g:
        g.write("\t".join(cols) + "\n")
        g.write("\t".join("%.6f" % v for v in row) + "\n")

    die_w = [sum(block_w["B_%d" % (d * bpd + b)] for b in range(bpd)) for d in range(DIES)]   # stack 0
    acct = dict(
        geometry=args.geom, columns=len(cols), hbm=args.hbm, organization=org, blocks_per_die=bpd,
        simulated_pcs=meta_pc, pcs_per_stack=n_pc_stack, pcs_covered=covered,
        replicate=args.replicate, shared_policy=args.shared_policy,
        dram_scale_applied=scale,
        per_block_W={k: block_w[k] for k in bank_cols},
        per_die_W_bottom_to_top=die_w,
        dram_stack_W=dram_stack_w, geom_stacks=n_geom_stacks,
        dram_package_W=dram_stack_w * args.stacks, stacks=args.stacks,
        phy_W_per_stack=args.phy_w, gpu_W=args.gpu_w,
        pim_W=args.pim_w, pim_target=args.pim_target,
        ptrace_total_W=sum(row),
        assumptions=[
            "channel c on die c // (channels/8); (ch,pc,bg,bank) enumerated in order and split evenly over 4 blocks/die",
            "replicate=%s: %s" % (args.replicate, "simulated PC(s) represent every PC" if args.replicate == "uniform"
                                  else "uncovered PCs get 0 W"),
            "shared (non-bank) energy folded into banks: %s" % args.shared_policy,
            "package = %d identical stacks; only one stack is in the floorplan" % args.stacks,
            "host-side PHY/controller power is an input (--phy-w), not simulated",
            "sid_folded: ramulator's 2 ranks (stack IDs) per PC are folded into one 16-bank PC; die placement ignores SID",
        ],
    )
    with open(args.out + ".map.json", "w") as g:
        json.dump(acct, g, indent=2)
    print("[bridge] %s: %d cols | DRAM stack %.2f W (x%d = %.1f W pkg) | GPU %.1f W | PHY %.1f W | PIM %.1f W -> %s"
          % (os.path.basename(args.out), len(cols), dram_stack_w, args.stacks, dram_stack_w * args.stacks,
             args.gpu_w, args.phy_w, args.pim_w, out_pt))
    lo, hi = min(block_w.values()), max(block_w.values())
    print("[bridge] per-block W: min %.4f max %.4f (spread %.1f%%) | per-die W: %s"
          % (lo, hi, (hi / lo - 1) * 100 if lo > 0 else float("nan"),
             " ".join("%.3f" % d for d in die_w)))


if __name__ == "__main__":
    main()
