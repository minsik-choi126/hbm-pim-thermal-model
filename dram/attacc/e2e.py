"""AttAcc's own end-to-end simulator (main.py): H100, LLaMA-65B, 8 GPUs (TP 8), batch 94 (the KV-capacity batch at
context 2048), three systems:
    gpu_only           dgx                                         attention on the GPU
    pim_alternating    dgx-attacc --pim bank --powerlimit          attention on bank-level PIM after the FC (no overlap)
    pim_head_pipeline  dgx-attacc --pim bank --powerlimit --pipeopt   AttAcc's head-level pipeline
Default: one decode step at context 2048 (--lin 2047 --lout 2). --all adds AttAcc's default (--lin 2048 --lout 128,
contexts 2049-2175 averaged; about 12 s per context under load, so about 25 min).
main.py runs in a temporary copy of the simulator tree (it writes output.csv and its ramulator.out cache there) with a
`python` shim on PATH (its trace generator is called as "python"). Interpreter: ATTACC_PYTHON, else the venv made by
dram/setup/attacc.sh, else this one (needs numpy and pandas).

python -m dram.attacc.e2e [--all]       # writes dram/results/attacc_e2e_h100_llama65b.csv
"""
import csv
import os
import shutil
import subprocess
import sys
import tempfile

from dram import common as C

RUNS = (("gpu_only", "dgx", False), ("pim_alternating", "dgx-attacc", False), ("pim_head_pipeline", "dgx-attacc", True))
CONTEXTS = ((2048, 128), (2047, 2))          # (lin, lout)
GPU, MODEL, NGPU, BATCH = "H100", "LLAMA-65B", 8, 94
MS = {"step_ms": "g_time (ms)", "fc_ms": "g_fc", "attn_ms": "g_matmul", "comm_ms": "g_comm", "etc_ms": "g_etc",
      "softmax_ms": "g_softmax"}
NJ = {"energy_J": ("g_energy (nJ)",), "dram_energy_J": ("g_dram_energy",), "fc_mem_J": ("g_fc_mem_energy",),
      "fc_comp_J": ("g_fc_comp_energy",), "attn_mem_J": ("g_attn_mem_energy",), "attn_comp_J": ("g_attn_comp_energy",),
      "etc_J": ("g_etc_mem_energy", "g_etc_comp_energy"), "comm_J": ("g_comm_energy",)}
OUT = "attacc_e2e_h100_llama65b.csv"


def python():
    venv = os.path.join(C.ATTACC, ".venv", "bin", "python")
    return os.environ.get("ATTACC_PYTHON") or (venv if os.path.exists(venv) else sys.executable)


def main():
    C.need(C.RAMULATOR, "attacc")
    py = python()
    ctxs = CONTEXTS if "--all" in sys.argv[1:] else CONTEXTS[1:]
    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        work = os.path.join(tmp, "attacc")
        shutil.copytree(C.ATTACC, work, symlinks=True, ignore=shutil.ignore_patterns(
            ".git", ".venv", "build", "__pycache__", "output.csv", "ramulator.out", "ext", "perf_comparison", "rh_study",
            "verilog_verification", "resources"))
        shim = os.path.join(tmp, "bin")
        os.makedirs(shim)
        with open(os.path.join(shim, "python"), "w") as f:
            f.write(f'#!/bin/sh\nexec "{py}" "$@"\n')
        os.chmod(os.path.join(shim, "python"), 0o755)
        env = dict(os.environ, PATH=shim + os.pathsep + os.environ["PATH"], PYTHONPATH=work)
        for lin, lout in ctxs:
            for run, system, pipe in RUNS:
                cmd = [py, "main.py", "--system", system, "--gpu", GPU, "--ngpu", str(NGPU), "--model", MODEL,
                       "--lin", str(lin), "--lout", str(lout), "--batch", str(BATCH)]
                if system == "dgx-attacc":
                    cmd += ["--pim", "bank", "--powerlimit"] + (["--pipeopt"] if pipe else [])
                p = subprocess.run(cmd, cwd=work, env=env, capture_output=True, text=True)
                if p.returncode:
                    raise SystemExit(f"{' '.join(cmd)} failed:\n{p.stderr[-2000:]}")
                with open(os.path.join(work, "output.csv")) as f:
                    r = next(csv.DictReader(f))
                row = dict(run=run, attacc_system=system, pipeopt=pipe, gpu=GPU, model=r["model"], Lin=lin, Lout=lout,
                           batch=BATCH)
                row.update({k: float(r[v]) for k, v in MS.items()})
                row.update({k: sum(float(r[c]) for c in cs) / 1e9 for k, cs in NJ.items()})
                rows.append(row)
    with open(C.out(OUT), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    for r in rows:
        print(f"{r['run']:18s} lin {r['Lin']} lout {r['Lout']}: step {r['step_ms']:.2f} ms, energy {r['energy_J']:.2f} J")


if __name__ == "__main__":
    main()
