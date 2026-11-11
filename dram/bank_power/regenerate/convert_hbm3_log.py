#!/usr/bin/env python3
"""LLMSimulator/Ramulator2 HBM3 TraceRecorder log -> DRAMPower 8-column CSV (HBM3_runner format).

Raw input (cmd_hbm3.log.ch0), 13 comma+space columns:
    clk, CMD, channel, pseudochannel, rank, bankgroup, bank, row, column, addr, pim_cmd, operand, req_type
Output, per pseudo-channel AND combined, CRLF, header:
    timestamp,command,channel_id,pseudochannel_id,bankgroup_id,bank_id,row_id,column_id

Differences from HBM-Power's drampower_convert.py (which was written for the HBM2 A100 log):
  * BOTH pseudo-channels are kept (one file each + a combined file); figure14's reference traces
    kept only PC0.
  * REFab and PREA are KEPT (drampower_convert.py never saw them: the HBM2 config ran no_refresh).
    REFab -> "REFA" (the CmdType string DRAMPower parses); PREA stays "PREA". Both are rank-level:
    bankgroup/bank are written as 0 so the parser's flat-bank index is valid.
  * column is blanked on non-column commands, exactly as upstream.
  * trailing all-bank PRE flush per PC (same as upstream) so no interval is left open.
    python3 convert_hbm3_log.py raw/x_cmd_hbm3.log.ch0 out_prefix
"""
import sys

HDR = "timestamp,command,channel_id,pseudochannel_id,bankgroup_id,bank_id,row_id,column_id"
RENAME = {"REFab": "REFA", "REFAB": "REFA", "REF": "REFA"}


def convert(raw, prefix, rank_as_bg=False):
    per = {}
    open_row = {}
    maxclk = {}
    counts = {}
    with open(raw) as fi:
        for line in fi:
            p = [x.strip() for x in line.split(",")]
            if len(p) < 9:
                continue
            clk, cmd, ch, pch, _rank, bg, bank, row, col = p[:9]
            cmd = RENAME.get(cmd, cmd)
            if rank_as_bg and cmd not in ("REFab", "REFAB", "REF", "REFA", "PREA") and _rank not in ("", "-1"):
                bg = str(int(_rank) * 4 + int(bg))          # keep the stack-ID (rank) as extra bank-group bits
            counts[cmd] = counts.get(cmd, 0) + 1
            if cmd in ("REFA", "PREA"):
                bg, bank, row, colf = "0", "0", "0", ""
            else:
                colf = col if cmd in ("RD", "WR", "RDA", "WRA") else ""
            key = (ch, pch)
            per.setdefault(key, []).append((clk, cmd, ch, pch, bg, bank, row, colf))
            if cmd == "ACT":
                open_row[(key, bg, bank)] = row
            c = int(clk)
            if c > maxclk.get(key, 0):
                maxclk[key] = c
    outs = []
    for key in sorted(per):
        rows = per[key]
        idx = 0
        for bg in range(8 if rank_as_bg else 4):
            for bank in range(4):
                idx += 1
                r = open_row.get((key, str(bg), str(bank)), "0")
                rows.append((str(maxclk[key] + 4 * idx), "PRE", key[0], key[1], str(bg), str(bank), r, ""))
        out = "%s_ch%s_pc%s.csv" % (prefix, key[0], key[1])
        with open(out, "w", newline="") as fo:
            fo.write(HDR + "\r\n")
            for r in rows:
                fo.write(",".join(r) + "\r\n")
        outs.append((out, len(rows)))
    allrows = sorted((r for k in per for r in per[k]), key=lambda r: (int(r[0]), r[3]))
    comb = prefix + "_all.csv"
    with open(comb, "w", newline="") as fo:
        fo.write(HDR + "\r\n")
        for r in allrows:
            fo.write(",".join(r) + "\r\n")
    print("%s -> %s (%d rows) + %s ; cmds: %s" % (raw, comb, len(allrows), ", ".join("%s(%d)" % o for o in outs),
                                                 " ".join("%s=%d" % kv for kv in sorted(counts.items()))))


if __name__ == "__main__":
    convert(sys.argv[1], sys.argv[2], rank_as_bg=("--rank-as-bg" in sys.argv))
