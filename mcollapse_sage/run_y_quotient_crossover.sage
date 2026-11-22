"""Exploratory scale audit for the exact y-fiber MCOLLAPSE lookup crossover."""

import hashlib
import json
import os
from sage.env import SAGE_VERSION

load("mcollapse_sage/rank_guided_mcollapse.sage")
load("mcollapse_sage/symmetry_quotient_mcollapse.sage")
load("mcollapse_sage/circuit_mcollapse.sage")
load("mcollapse_sage/y_quotient_mcollapse.sage")


def json_safe(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    try:
        if value in ZZ:
            return int(value)
    except (TypeError, ValueError):
        pass
    return value


configurations = ((4, 1201), (6, 1213), (8, 1217), (10, 1223), (12, 1229))
target_count = 32
variable_budget = 500000
cells = []

for base_size, seed_value in configurations:
    fixture = find_cm_fixture(13, seed_value)
    reference = PreparedQueryOrientedSymmetryQuotientMCollapse(fixture, base_size)
    candidate = PreparedYQuotientMCollapse(fixture, base_size)
    Q, targets = make_distinct_random_public_targets(
        fixture, target_count, seed_value + 19001
    )
    reference_run = run_circuit_mcollapse(
        reference, targets, variable_budget, seed_value + 19003, round_robin=True
    )
    candidate_run = run_circuit_mcollapse(
        candidate, targets, variable_budget, seed_value + 19003, round_robin=True
    )
    reference_result = reference_run.as_dict()
    candidate_result = candidate_run.as_dict()
    exact = bool(
        reference_run.solved
        and candidate_run.solved
        and ZZ(reference_run.recovered) * fixture.G == Q
        and ZZ(candidate_run.recovered) * fixture.G == Q
    )
    traces_match = bool(
        reference_run.state.candidate_trace == candidate_run.state.candidate_trace
    )
    online_ratio = RR(candidate_result["variable_work"]) / RR(
        reference_result["variable_work"]
    )
    cell = {
        "fixture": fixture.as_dict(),
        "base_size": int(base_size),
        "exact": exact,
        "candidate_traces_match": traces_match,
        "anchor_ranks_match": bool(reference_run.basis.rank == candidate_run.basis.rank),
        "reference": reference_result,
        "candidate": candidate_result,
        "online_work_ratio": float(online_ratio),
        "total_work_ratio": float(
            RR(candidate_result["total_with_preparation"])
            / RR(reference_result["total_with_preparation"])
        ),
        "y_bucket_count": int(len(candidate.quotient_index.index)),
        "mean_rows_per_y_bucket": float(
            RR(candidate.quotient_index.representative_count)
            / RR(len(candidate.quotient_index.index))
        ),
    }
    if not exact or not traces_match or not cell["anchor_ranks_match"]:
        raise ArithmeticError("y crossover audit changed exact Circuit behavior")
    cells.append(cell)
    print("%s m=%s n=%s buckets=%s online=%.6f total=%.6f" % (
        fixture.label,
        base_size,
        fixture.n,
        cell["y_bucket_count"],
        cell["online_work_ratio"],
        cell["total_work_ratio"],
    ))

summary = {
    "experiment": "Exploratory exact y-fiber MCOLLAPSE bucket-density crossover audit",
    "sage_version": str(SAGE_VERSION),
    "preregistered": False,
    "base_sizes": [int(item[0]) for item in configurations],
    "target_count": target_count,
    "cells": cells,
    "all_exact": bool(all(cell["exact"] for cell in cells)),
    "all_traces_match": bool(all(cell["candidate_traces_match"] for cell in cells)),
    "online_win_cells": int(sum(cell["online_work_ratio"] < 1 for cell in cells)),
    "verdict": "MEASURE_Y_QUOTIENT_ON_LARGER_CLASSIC_FIXTURES",
}
output_directory = "mcollapse_sage/results"
os.makedirs(output_directory, exist_ok=True)
output_path = os.path.join(output_directory, "y_quotient_crossover_summary.json")
with open(output_path, "w", encoding="utf-8") as output_file:
    json.dump(json_safe(summary), output_file, indent=2, sort_keys=True)
    output_file.write("\n")
with open(output_path, "rb") as input_file:
    digest = hashlib.sha256(input_file.read()).hexdigest()
print("wrote %s" % output_path)
print("sha256: %s" % digest)
