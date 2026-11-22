"""Run the preregistered public-density adaptive y/point MCOLLAPSE index."""

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


held_out = ((8, 1301), (9, 1303), (10, 1307), (11, 1319), (12, 1321), (13, 1327))
base_size = 12
target_count = 32
variable_budget = 250000
cells = []

for subgroup_bits, seed_value in held_out:
    fixture = find_cm_fixture(subgroup_bits, seed_value)
    reference = PreparedQueryOrientedSymmetryQuotientMCollapse(fixture, base_size)
    candidate = PreparedAdaptiveYPointMCollapse(fixture, base_size)
    Q, targets = make_distinct_random_public_targets(
        fixture, target_count, seed_value + 23001
    )
    reference_run = run_circuit_mcollapse(
        reference, targets, variable_budget, seed_value + 23003, round_robin=True
    )
    candidate_run = run_circuit_mcollapse(
        candidate, targets, variable_budget, seed_value + 23003, round_robin=True
    )
    reference_result = reference_run.as_dict()
    candidate_result = candidate_run.as_dict()
    exact = bool(
        reference_run.solved
        and candidate_run.solved
        and ZZ(reference_run.recovered) * fixture.G == Q
        and ZZ(candidate_run.recovered) * fixture.G == Q
        and reference_run.recovered == candidate_run.recovered
    )
    traces_match = bool(
        reference_run.state.candidate_trace == candidate_run.state.candidate_trace
    )
    ranks_match = bool(reference_run.basis.rank == candidate_run.basis.rank)
    online_ratio = RR(candidate_result["variable_work"]) / RR(
        reference_result["variable_work"]
    )
    total_ratio = RR(candidate_result["total_with_preparation"]) / RR(
        reference_result["total_with_preparation"]
    )
    cell = {
        "fixture": fixture.as_dict(),
        "selected_layout": candidate.as_dict()["selected_layout"],
        "predicted_pair_orbits": int(candidate.predicted_pair_orbits),
        "exact": exact,
        "candidate_traces_match": traces_match,
        "anchor_ranks_match": ranks_match,
        "reference": reference_result,
        "candidate": candidate_result,
        "online_work_ratio": float(online_ratio),
        "total_work_ratio": float(total_ratio),
    }
    if not exact or not traces_match or not ranks_match:
        raise ArithmeticError("adaptive index changed exact Circuit behavior")
    cells.append(cell)
    print("%s layout=%s online=%.6f total=%.6f" % (
        fixture.label,
        cell["selected_layout"],
        cell["online_work_ratio"],
        cell["total_work_ratio"],
    ))

y_cells = [cell for cell in cells if cell["selected_layout"] == "y-fiber"]
point_cells = [
    cell for cell in cells if cell["selected_layout"] == "query-oriented-point"
]
all_correct = all(
    cell["exact"]
    and cell["candidate_traces_match"]
    and cell["anchor_ranks_match"]
    and cell["reference"]["verification_failures"] == 0
    and cell["candidate"]["verification_failures"] == 0
    for cell in cells
)
reference_total = sum(ZZ(cell["reference"]["total_with_preparation"]) for cell in cells)
candidate_total = sum(ZZ(cell["candidate"]["total_with_preparation"]) for cell in cells)
passed = bool(
    all_correct
    and y_cells
    and point_cells
    and all(cell["online_work_ratio"] <= 1 for cell in cells)
    and all(cell["total_work_ratio"] <= 1 for cell in cells)
    and all(cell["online_work_ratio"] < 1 for cell in y_cells)
    and candidate_total < reference_total
)
summary = {
    "experiment": "Preregistered adaptive y/point MCOLLAPSE pair index",
    "sage_version": str(SAGE_VERSION),
    "preregistered": True,
    "preregistration": "ADAPTIVE_Y_MCOLLAPSE_PREREGISTRATION.md",
    "base_size": base_size,
    "target_count": target_count,
    "aggregate": {
        "all_correctness_checks_passed": bool(all_correct),
        "y_selected_cells": len(y_cells),
        "point_selected_cells": len(point_cells),
        "reference_total_work": int(reference_total),
        "candidate_total_work": int(candidate_total),
        "aggregate_total_work_ratio": float(RR(candidate_total) / RR(reference_total)),
    },
    "promotion_gate": {
        "passed": passed,
        "verdict": (
            "PROMOTE_ADAPTIVE_Y_POINT_INDEX"
            if passed
            else "KEEP_POINT_QUERY_ORIENTED_INDEX"
        ),
    },
    "cells": cells,
}
output_directory = "mcollapse_sage/results"
os.makedirs(output_directory, exist_ok=True)
output_path = os.path.join(output_directory, "adaptive_y_mcollapse_summary.json")
with open(output_path, "w", encoding="utf-8") as output_file:
    json.dump(json_safe(summary), output_file, indent=2, sort_keys=True)
    output_file.write("\n")
with open(output_path, "rb") as input_file:
    digest = hashlib.sha256(input_file.read()).hexdigest()
print("promotion gate: %s" % summary["promotion_gate"]["verdict"])
print("wrote %s" % output_path)
print("sha256: %s" % digest)
