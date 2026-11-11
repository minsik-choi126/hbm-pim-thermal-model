"""DRAM power map of one HBM3 stack from the per-bank DRAMPower energy, without a floorplan.

Same accounting as bank_power_to_ptrace.py (shared energy in proportion to bank energy, the two simulated pseudo
channels replicated over the 32 of the stack, 4 blocks per die): per pseudo channel, per block and per die W, and the
stack total. per_block_W lists the 32 blocks B_0..B_31 of one stack (die-major, 4 per die), the DRAM floor
the thermal model puts on every stack.

python -m dram.bank_power.stack_power      # writes dram/results/bank_power_stack.json
"""
import glob
import json
import os

from dram import common as C
from dram.bank_power import bank_power_to_ptrace as BR

DATA = sorted(glob.glob(os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "*_weights.json")))
BLOCKS_PER_DIE = 4


def stack_map(paths=DATA, shared_policy="proportional"):
    org = BR.ORG["hbm3"]
    recs = BR.load_pc_entries(paths)
    sim, pcs = {}, []
    for rec in recs:
        w, w_shared, dur = BR.pc_bank_watts(rec, shared_policy)
        key = (int(rec.get("channel", 0)), int(rec.get("pseudochannel", 0)))
        sim[key] = w
        pcs.append(dict(src=rec["_src"], channel=key[0], pseudochannel=key[1], W=sum(w), shared_W=w_shared,
                        duration_us=dur * 1e6))
    n = org["bg"] * org["bank"]
    mean_pc = [sum(v[i] for v in sim.values()) / len(sim) for i in range(n)]
    block = [[0.0] * BLOCKS_PER_DIE for _ in range(BR.DIES)]
    for ch in range(org["ch"]):
        for pc in range(org["pc"]):
            w = sim.get((ch, pc), mean_pc)
            for bg in range(org["bg"]):
                for bank in range(org["bank"]):
                    die, blk = BR.block_of(org, ch, pc, bg, bank, BLOCKS_PER_DIE)
                    block[die][blk] += w[bg * org["bank"] + bank]
    flat = [x for d in block for x in d]
    return dict(inputs=[os.path.basename(p) for p in paths], shared_policy=shared_policy, simulated_pcs=pcs,
                stack_W=sum(flat), per_block_W=flat, block_W=dict(min=min(flat), max=max(flat), n=len(flat)),
                block_spread_pct=100 * (max(flat) / min(flat) - 1), die_W_bottom_to_top=[sum(d) for d in block])


def main():
    m = stack_map()
    with open(C.out("bank_power_stack.json"), "w") as f:
        json.dump(m, f, indent=1)
    print(f"DRAM stack {m['stack_W']:.3f} W: {m['block_W']['n']} blocks of {m['block_W']['min']:.5f}-{m['block_W']['max']:.5f} W "
          f"(spread {m['block_spread_pct']:.4f} %), per die {m['die_W_bottom_to_top'][0]:.3f} W")


if __name__ == "__main__":
    main()
