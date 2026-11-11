"""Column order of a HotSpot power trace for a grid-model layer file (.lcf).

HotSpot's grid model with detailed_3D maps power by position: the ptrace columns are the concatenation of the block
names of every powered layer (Power = Y), in LCF order and in floorplan-file order. Layers without power (interposer,
TIM) contribute no columns.
"""
import os


def powered_layer_flps(lcf):
    """[flp path, ...] of the powered layers in LCF order (each layer is 7 non-comment lines:
    number, lateral heat flow, power, specific heat, resistivity, thickness, floorplan)."""
    hs = os.path.dirname(lcf)
    recs = [s.strip() for s in open(lcf) if s.strip() and not s.strip().startswith("#")]
    if len(recs) == 0 or len(recs) % 7:
        raise SystemExit(f"malformed LCF {lcf}: {len(recs)} non-comment records is not a positive multiple of 7")
    flps = []
    for i in range(0, len(recs), 7):
        if recs[i + 2].upper() == "Y":
            path = os.path.join(hs, os.path.basename(recs[i + 6]))
            if not os.path.isfile(path):
                raise SystemExit(f"powered-layer floorplan missing: {path}")
            flps.append(path)
    return flps


def block_names_in_order(flp):
    return [s.split()[0] for s in (ln.strip() for ln in open(flp)) if s and not s.startswith("#")]


def build_columns(lcf):
    cols = []
    for flp in powered_layer_flps(lcf):
        cols.extend(block_names_in_order(flp))
    return cols
