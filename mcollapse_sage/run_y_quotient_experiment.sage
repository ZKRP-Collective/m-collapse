"""Run the preregistered y-fiber MCOLLAPSE pair-index comparison."""

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


def exact_median(values):
    ordered = sorted(values)
    midpoint = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[midpoint]
    return (ordered[midpoint - 1] + ordered[midpoint]) / 2


held_out = ((8, 907), (9, 911), (10, 919), (11, 929), (12, 937), (13, 941))
base_size = 12
target_count = 32
variable_budget = 250000
cells = []

for subgroup_bits, seed_value in held_out:
    fixture = find_cm_fixture(subgroup_bits, seed_value)
    reference = PreparedQueryOrientedSymmetryQuotientMCollapse(fixture, base_size)
    candidate = PreparedYQuotientMCollapse(fixture, base_size)
    audit = audit_y_lookup_equivalence(reference, candidate)
    Q, targets = make_distinct_random_public_targets(
        fixture, target_count, seed_value + 15101
    )
    reference_run = run_circuit_mcollapse(
        reference, targets, variable_budget, seed_value + 17107, round_robin=True
    )
    candidate_run = run_circuit_mcollapse(
        candidate, targets, variable_budget, seed_value + 17107, round_robin=True
    )
    reference_result = reference_run.as_dict()
    candidate_result = candidate_run.as_dict()
    reference_online = ZZ(reference_result["variable_work"])
    candidate_online = ZZ(candidate_result["variable_work"])
    reference_total = ZZ(reference_result["total_with_preparation"])
    candidate_total = ZZ(candidate_result["total_with_preparation"])
    recovered_exactly = bool(
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
    cell = {
        "fixture": fixture.as_dict(),
        "target_count": target_count,
        "distinct_target_points": len({point_key(target.point) for target in targets}),
        "reference_prepared": reference.as_dict(),
        "candidate_prepared": candidate.as_dict(),
        "lookup_audit": audit,
        "reference": reference_result,
        "candidate": candidate_result,
        "recovered_exactly": recovered_exactly,
        "candidate_traces_match": traces_match,
        "anchor_ranks_match": ranks_match,
        "online_work_ratio": float(RR(candidate_online) / RR(reference_online)),
        "total_work_ratio": float(RR(candidate_total) / RR(reference_total)),
    }
    cells.append(cell)
    print("%s rows=%s online=%.6f total=%.6f exact=%s" % (
        fixture.label,
        audit["exact_pair_rows_checked"],
        cell["online_work_ratio"],
        cell["total_work_ratio"],
        recovered_exactly,
    ))

all_correct = all(
    cell["distinct_target_points"] == target_count
    and cell["lookup_audit"]["row_set_mismatches"] == 0
    and cell["lookup_audit"]["group_failures"] == 0
    and cell["recovered_exactly"]
    and cell["candidate_traces_match"]
    and cell["anchor_ranks_match"]
    and cell["reference"]["verification_failures"] == 0
    and cell["candidate"]["verification_failures"] == 0
    and cell["reference"]["exact_rank_audit"]["system_consistent"]
    and cell["candidate"]["exact_rank_audit"]["system_consistent"]
    for cell in cells
)
online_ratios = [RR(cell["online_work_ratio"]) for cell in cells]
total_ratios = [RR(cell["total_work_ratio"]) for cell in cells]
online_wins = sum(ratio < 1 for ratio in online_ratios)
median_online = exact_median(online_ratios)
median_total = exact_median(total_ratios)
passed = bool(
    all_correct
    and online_wins >= 5
    and median_online <= RR("0.90")
    and median_total <= RR("0.98")
    and max(total_ratios) <= 1
)

summary = {
    "experiment": "SageMath preregistered y-fiber MCOLLAPSE pair index",
    "sage_version": str(SAGE_VERSION),
    "preregistered": True,
    "preregistration": "Y_QUOTIENT_MCOLLAPSE_PREREGISTRATION.md",
    "base_size": base_size,
    "target_count": target_count,
    "held_out_cells": len(held_out),
    "aggregate": {
        "all_correctness_checks_passed": bool(all_correct),
        "online_work_win_cells": int(online_wins),
        "median_online_work_ratio": float(median_online),
        "worst_online_work_ratio": float(max(online_ratios)),
        "median_total_work_ratio": float(median_total),
        "worst_total_work_ratio": float(max(total_ratios)),
        "subgroup_points_checked": int(sum(
            cell["lookup_audit"]["subgroup_points_checked"] for cell in cells
        )),
        "exact_pair_rows_checked": int(sum(
            cell["lookup_audit"]["exact_pair_rows_checked"] for cell in cells
        )),
    },
    "promotion_gate": {
        "passed": passed,
        "verdict": (
            "PROMOTE_Y_QUOTIENT_PAIR_INDEX"
            if passed
            else "KEEP_POINT_QUERY_ORIENTED_INDEX"
        ),
    },
    "cells": cells,
}

output_directory = "mcollapse_sage/results"
os.makedirs(output_directory, exist_ok=True)
output_path = os.path.join(output_directory, "y_quotient_summary.json")
with open(output_path, "w", encoding="utf-8") as output_file:
    json.dump(json_safe(summary), output_file, indent=2, sort_keys=True)
    output_file.write("\n")
with open(output_path, "rb") as input_file:
    digest = hashlib.sha256(input_file.read()).hexdigest()

print("promotion gate: %s" % summary["promotion_gate"]["verdict"])
print("wrote %s" % output_path)
print("sha256: %s" % digest)
