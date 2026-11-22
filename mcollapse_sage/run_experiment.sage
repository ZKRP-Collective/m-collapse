import json
import os
from sage.env import SAGE_VERSION

load("mcollapse_sage/rank_guided_mcollapse.sage")


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


fixture_requests = ((7, 11), (7, 19), (8, 23), (8, 31), (9, 37), (9, 43))
frontiers = (250, 500, 750, 1000, 2000)
primary_frontier = 750
cells = []

for subgroup_bits, seed_value in fixture_requests:
    fixture = find_cm_fixture(subgroup_bits, seed_value)
    orbit_count = (fixture.n - 1) // 6
    base_size = min(ZZ(12), orbit_count)
    cell = compare_policies(
        fixture=fixture,
        base_size=base_size,
        target_count=64,
        variable_budget=max(frontiers),
        frontiers=frontiers,
        seed_value=seed_value + 701,
        primary_frontier=primary_frontier,
    )
    cells.append(cell)
    print(
        "%s: n=%s base=%s guided/baseline@%s=%s"
        % (
            fixture.label,
            fixture.n,
            base_size,
            primary_frontier,
            cell["primary_guided_to_baseline_yield_ratio"],
        )
    )

ratios = [
    RR(cell["primary_guided_to_baseline_yield_ratio"])
    for cell in cells
    if cell["primary_guided_to_baseline_yield_ratio"] is not None
]
passes = len(ratios) == len(cells) and all(ratio > RR("1.10") for ratio in ratios)
ordered_ratios = sorted(ratios)
median_ratio = exact_median(ordered_ratios)
filtered_ratios = [RR(cell["primary_filtered_to_baseline_yield_ratio"]) for cell in cells]
guided_to_saturated_ratios = [
    RR(cell["primary_guided_to_saturated_yield_ratio"]) for cell in cells
]
verification_failures = sum(
    cell["policies"][policy]["verification_failures"]
    for cell in cells
    for policy in ("weighted-baseline", "rank-filtered", "rank-saturated", "rank-guided")
)
def full_at_max(policy):
    return sum(
        cell["policies"][policy]["frontiers"][-1]["independent_relations"]
        == cell["prepared"]["matrix_width"]
        for cell in cells
    )
summary = {
    "experiment": "SageMath-native online rank-guided weighted MCOLLAPSE",
    "sage_version": str(SAGE_VERSION),
    "arithmetic": "Sage GF(n), EllipticCurve(GF(p)), vector, and matrix.rank",
    "fixture_count": len(cells),
    "frontiers": [int(value) for value in frontiers],
    "primary_frontier": int(primary_frontier),
    "aggregate": {
        "guided_to_baseline_ratio_min": float(min(ordered_ratios)),
        "guided_to_baseline_ratio_median": float(median_ratio),
        "guided_to_baseline_ratio_max": float(max(ordered_ratios)),
        "guided_strictly_better_fixtures": int(sum(ratio > 1 for ratio in ratios)),
        "guided_above_1_10_fixtures": int(sum(ratio > RR("1.10") for ratio in ratios)),
        "filtered_to_baseline_ratio_median": float(exact_median(filtered_ratios)),
        "guided_to_saturated_ratio_median": float(exact_median(guided_to_saturated_ratios)),
        "full_rank_at_max_frontier": {
            policy: int(full_at_max(policy))
            for policy in ("weighted-baseline", "rank-filtered", "rank-saturated", "rank-guided")
        },
        "verification_failures": int(verification_failures),
    },
    "promotion_gate": {
        "criterion": "guided/baseline independent-relations-per-primitive > 1.10 on every fixture at the primary frontier",
        "passed": bool(passes),
        "verdict": "PROMOTE" if passes else "DO_NOT_PROMOTE",
    },
    "cells": cells,
}

output_directory = "mcollapse_sage/results"
os.makedirs(output_directory, exist_ok=True)
output_path = os.path.join(output_directory, "rank_guided_summary.json")
with open(output_path, "w", encoding="utf-8") as output_file:
    json.dump(json_safe(summary), output_file, indent=2, sort_keys=True)
    output_file.write("\n")

print("wrote %s" % output_path)
print("promotion gate: %s" % summary["promotion_gate"]["verdict"])
