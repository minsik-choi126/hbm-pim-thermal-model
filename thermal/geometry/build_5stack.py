#!/usr/bin/env python3
"""H100-like 2.5D package as a HotSpot grid-model layer stack (stack.lcf) and floorplans (*.flp).

Layout (mm, x to the right, y up): six HBM stack sites, three in a column left of the GPU die and three right of it,
each 8.0 x 12.19, 0.5 mm from the GPU die (28.53 x 28.53 mm, 814 mm^2) centred on the stack columns. Five sites are
powered and the sixth (top right) is a dark site, as on H100 SXM.

Layers (bottom to top): interposer, bond, GPU die + HBM base dies, bond, then 8 x (DRAM die, bond).
Block names: C_0 (GPU die), LC_<s>_<i> (base-die blocks of stack s, 2 x 2), B_<n> with n = 32 s + 4 die + i
(DRAM-die blocks, 2 x 2 per die), D_* / DLC_* (dark site, never powered), X_* (filler between the dies).

    python3 thermal/geometry/build_5stack.py --out thermal/geometry/h100_5stack
"""
import argparse
import os

W_ST, H_ST, GAP, G = 8.0, 12.19, 0.5, 28.53
SITES = [(0.0, i * H_ST) for i in range(3)] + [(W_ST + GAP + G + GAP, i * H_ST) for i in range(3)]
BOX_W, BOX_H = 2 * W_ST + 2 * GAP + G, 3 * H_ST
GY0 = (BOX_H - G) / 2
GX0 = W_ST + GAP
FILL = "2875000\t0.13"       # filler: specific heat J/(m^3 K), resistivity m K / W
LAYER_MAT = dict(si=("1750000.0", "0.007", "5e-05"), tim=("4000000.0", "0.25", "2e-05"))


def rect(name, w, h, x, y, mat=None):
    s = "%s\t%.6f\t%.6f\t%.6f\t%.6f" % (name, w / 1000, h / 1000, x / 1000, y / 1000)
    return s + ("\t" + mat if mat else "")


def fillers_between(prefix, core_layer):
    """Cover everything outside the six sites: gap columns, and the GPU column (split around the GPU die on the
    GPU-die layer, whole on the DRAM-die layers)."""
    out = [rect(prefix + "g0", GAP, BOX_H, W_ST, 0, FILL), rect(prefix + "g1", GAP, BOX_H, GX0 + G, 0, FILL)]
    if core_layer:
        out += [rect(prefix + "c0", G, GY0, GX0, 0, FILL), rect(prefix + "c1", G, BOX_H - GY0 - G, GX0, GY0 + G, FILL)]
    else:
        out += [rect(prefix + "c", G, BOX_H, GX0, 0, FILL)]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--powered", type=int, default=5, help="powered stack sites (the rest are dark)")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    hdr = "# Line Format: <unit-name>\t<width>\t<height>\t<left-x>\t<bottom-y>\t[<specific-heat-capacity>\t<thermal-resistivity>]"
    open(os.path.join(a.out, "interposer.flp"), "w").write(hdr + "\n" + rect("I_0", BOX_W, BOX_H, 0, 0) + "\n")
    open(os.path.join(a.out, "tim.flp"), "w").write(hdr + "\n" + rect("TB_0", BOX_W, BOX_H, 0, 0) + "\n")
    quad = [(0, 0), (W_ST / 2, 0), (0, H_ST / 2), (W_ST / 2, H_ST / 2)]
    core = [rect("C_0", G, G, GX0, GY0)]
    for s, (x0, y0) in enumerate(SITES):
        dark = s >= a.powered
        for i, (dx, dy) in enumerate(quad):
            core.append(rect(("DLC_%d_%d" if dark else "LC_%d_%d") % (s, i), W_ST / 2, H_ST / 2, x0 + dx, y0 + dy))
    core += fillers_between("X_", True)
    open(os.path.join(a.out, "core_and_mem_ctrl.flp"), "w").write(hdr + "\n" + "\n".join(core) + "\n")
    for die in range(8):
        rows = []
        for s, (x0, y0) in enumerate(SITES):
            dark = s >= a.powered
            for i, (dx, dy) in enumerate(quad):
                nm = ("D_%d_%d_%d" % (s, die, i)) if dark else ("B_%d" % (s * 32 + die * 4 + i))
                rows.append(rect(nm, W_ST / 2, H_ST / 2, x0 + dx, y0 + dy))
        rows += fillers_between("X_", False)
        open(os.path.join(a.out, "mem_bank_%d.flp" % (die + 1)), "w").write(hdr + "\n" + "\n".join(rows) + "\n")
    L = []

    def layer(n, lateral, power, mat, flp):
        c, r, t = LAYER_MAT[mat]
        L.append("# Layer %d\n%d\n%s\n%s\n%s\n%s\n%s\n%s\n" % (n, n, lateral, power, c, r, t, flp))
    layer(0, "Y", "N", "si", "interposer.flp")
    layer(1, "Y", "N", "tim", "tim.flp")
    layer(2, "Y", "Y", "si", "core_and_mem_ctrl.flp")
    layer(3, "Y", "N", "tim", "tim.flp")
    n = 4
    for die in range(8):
        layer(n, "Y", "Y", "si", "mem_bank_%d.flp" % (die + 1))
        layer(n + 1, "Y", "N", "tim", "tim.flp")
        n += 2
    open(os.path.join(a.out, "stack.lcf"), "w").write(
        "#File Format:\n#<Layer Number>\n#<Lateral heat flow Y/N?>\n#<Power Dissipation Y/N?>\n#<Specific heat capacity in "
        "J/(m^3K)>\n#<Resistivity in (m-K)/W>\n#<Thickness in m>\n#<floorplan file>\n\n" + "\n".join(L))
    print("box %.2f x %.2f mm, GPU %.2f^2 at (%.2f,%.2f), %d sites (%d powered) -> %s"
          % (BOX_W, BOX_H, G, GX0, GY0, len(SITES), a.powered, a.out))


if __name__ == "__main__":
    main()
