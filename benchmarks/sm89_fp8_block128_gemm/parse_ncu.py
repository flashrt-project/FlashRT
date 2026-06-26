#!/usr/bin/env python3
from __future__ import annotations

import csv
import sys
from pathlib import Path


METRICS = [
    "Duration",
    "Registers Per Thread",
    "Dynamic Shared Memory Per Block",
    "Block Limit Registers",
    "Block Limit Shared Mem",
    "Theoretical Occupancy",
    "Achieved Occupancy",
    "Achieved Active Warps Per SM",
    "Eligible Warps Per Scheduler",
    "No Eligible",
    "Issue Slots Busy",
    "SM Busy",
    "Compute (SM) Throughput",
    "Memory Throughput",
    "DRAM Throughput",
    "L1/TEX Cache Throughput",
    "L2 Cache Throughput",
    "L1/TEX Hit Rate",
    "L2 Hit Rate",
    "Warp Cycles Per Issued Instruction",
    "Executed Instructions",
    "Issued Instructions",
]


def main() -> int:
    if len(sys.argv) != 2:
        print(f"usage: {Path(sys.argv[0]).name} <ncu_details.csv>", file=sys.stderr)
        return 2

    values: dict[str, tuple[str, str]] = {}
    with open(sys.argv[1], newline="") as f:
        for row in csv.DictReader(f):
            name = row.get("Metric Name", "")
            if name in METRICS:
                values[name] = (row.get("Metric Value", ""), row.get("Metric Unit", ""))

    width = max(len(m) for m in METRICS)
    for metric in METRICS:
        value, unit = values.get(metric, ("", ""))
        print(f"{metric:<{width}}  {value:>12}  {unit}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
