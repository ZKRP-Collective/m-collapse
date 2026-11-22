"""Run the preregistered Circuit-MCOLLAPSE relation-count scaling gate."""

import hashlib
import json
import os
from sage.env import SAGE_VERSION

load("mcollapse_sage/circuit_mcollapse.sage")
load("mcollapse_sage/circuit_scaling_gate.sage")


def json_safe(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, float):
        return value
    try:
        if value in ZZ:
            return int(value)
    except (TypeError, ValueError):
        pass
    if value in QQ:
        return {"numerator": int(value.numerator()), "denominator": int(value.denominator())}
    try:
        return float(value)
    except (TypeError, ValueError):
        return value


widths = (8, 12, 16, 24, 32, 48, 64)
trials = 64
cells = []
for width in widths:
    cell = run_circuit_scaling_cell(width, trials, 31001 + width)
    cells.append(cell)
    print("m=%s median-rel=%s ratio=%.6f median-rank=%s" % (
        width,
        cell["median_relations"],
        float(cell["median_relations_over_m"]),
        cell["median_anchor_rank"],
    ))
slope = circuit_scaling_log_slope(cells)
all_exact = all(
    cell["exact_recoveries"] == trials and cell["solved"] == trials
    for cell in cells
)
reject_sublinear = bool(
    all_exact
    and cells[-1]["median_relations_over_m"] >= QQ(7) / 10
    and slope >= 0.85
)
summary = {
    "experiment": "Preregistered Circuit-MCOLLAPSE relation-count scaling",
    "sage_version": str(SAGE_VERSION),
    "preregistered": True,
    "preregistration": "CIRCUIT_SCALING_PREREGISTRATION.md",
    "scalar_field": "GF(secp256k1 subgroup order)",
    "support_size": 3,
    "trials_per_cell": trials,
    "cells": cells,
    "aggregate": {
        "all_exact": bool(all_exact),
        "log_log_relation_slope": slope,
    },
    "verdict": (
        "REJECT_SUBLINEAR_CIRCUIT_COUNT_HYPOTHESIS"
        if reject_sublinear
        else "RETAIN_SUBLINEAR_CIRCUIT_COUNT_HYPOTHESIS"
    ),
}
output_directory = "mcollapse_sage/results"
os.makedirs(output_directory, exist_ok=True)
output_path = os.path.join(output_directory, "circuit_scaling_summary.json")
with open(output_path, "w", encoding="utf-8") as output_file:
    json.dump(json_safe(summary), output_file, indent=2, sort_keys=True)
    output_file.write("\n")
with open(output_path, "rb") as input_file:
    digest = hashlib.sha256(input_file.read()).hexdigest()
print("slope: %.6f" % slope)
print("verdict: %s" % summary["verdict"])
print("wrote %s" % output_path)
print("sha256: %s" % digest)
