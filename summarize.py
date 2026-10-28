#!/usr/bin/env python3
"""Median (min-max) per session x unlock x undervolt x workload.

usage: summarize.py [runs.csv ...]   (default: results/*/runs.csv)
"""
import csv
import glob
import os
import statistics
import sys
from collections import defaultdict

UNITS = {"gemm": "TFLOPS", "sd15": "s/img", "zimage": "s/img", "vllm_prefill": "in tok/s"}


def unit(w):
    return UNITS.get(w) or UNITS.get(w.split("_")[0]) or "out tok/s"


def primary(w, r):
    thr = float(r["throughput"])
    return 1 / thr if unit(w) == "s/img" else thr


def fmt(xs, nd=2):
    return f"{statistics.median(xs):.{nd}f} ({min(xs):.{nd}f}-{max(xs):.{nd}f})"


def main():
    paths = sys.argv[1:] or sorted(glob.glob(os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "*", "runs.csv")))
    groups = defaultdict(list)
    for p in paths:
        with open(p) as f:
            for r in csv.DictReader(f):
                groups[(r["session"], r["unlock"], r["undervolt"], r["workload"])].append(r)
    print(f"{'session':12s} {'unl':3s} {'uv':3s} {'workload':16s} {'n':>2s}  {'primary median (min-max)':32s} "
          f"{'avg W':>6s} {'J/output':>10s} {'SM MHz':>6s} {'per MHz':>8s} {'peak C':>6s} {'fan':>4s} pass")
    for (sess, unl, uv, w), rows in sorted(groups.items()):
        prim = [primary(w, r) for r in rows]
        watts = [float(r["avg_w"]) for r in rows]
        # outputs per run = throughput x seconds (images, tokens or TFLOP)
        jpo = [float(r["joules"]) / (float(r["throughput"]) * float(r["seconds"])) for r in rows]
        mhz = [float(r["avg_sm_mhz"]) for r in rows]
        per_mhz = f"{statistics.median(float(r['throughput']) / float(r['avg_sm_mhz']) for r in rows):.4f}" if w.startswith("gemm") else ""
        passes = [r["pass"] for r in rows]
        verdict = "-" if not any(passes) else f"{passes.count('1')}/{len(passes)}" + (" nd" if "nd" in passes else "")
        print(f"{sess:12s} {unl:3s} {uv:3s} {w:16s} {len(rows):2d}  {fmt(prim) + ' ' + unit(w):32s} "
              f"{statistics.median(watts):6.0f} {statistics.median(jpo):10.3f} {statistics.median(mhz):6.0f} {per_mhz:>8s} "
              f"{max(float(r['peak_c']) for r in rows):6.0f} {max(float(r['peak_fan']) for r in rows):4.0f} {verdict}")


if __name__ == "__main__":
    main()
