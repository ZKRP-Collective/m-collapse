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


def compact_result(result):
    totals = empty_stats()
    for values in result["template_stats"].values():
        for name in totals:
            totals[name] += values[name]
    result["template_totals"] = {name: int(value) for name, value in totals.items()}
    del result["template_stats"]
    del result["shape_stats"]
    return result


held_out = (
    (7, 101, 12),
    (7, 107, 12),
    (8, 109, 12),
    (8, 113, 12),
    (9, 127, 12),
    (9, 131, 12),
    (9, 137, 24),
    (9, 139, 24),
    (10, 149, 24),
    (10, 151, 24),
)
policy_specs = (
    ("rank-saturated", "rank-saturated", 3),
    ("dual-saturated", "dual-saturated", 3),
    ("hybrid-d1", "hybrid-saturated", 1),
    ("hybrid-d2", "hybrid-saturated", 2),
    ("hybrid-d3", "hybrid-saturated", 3),
    ("hybrid-d4", "hybrid-saturated", 4),
    ("adaptive-dual", "adaptive-dual", 3),
)
cells = []

for subgroup_bits, seed_value, base_size in held_out:
    fixture = find_cm_fixture(subgroup_bits, seed_value)
    prepared = PreparedWeightedMCollapse(fixture, base_size)
    Q, targets = make_public_targets(fixture, 64, seed_value + 1701)
    for target in targets:
        assert target.point == target.u * fixture.G + target.v * Q
    if base_size == 12:
        frontiers = (500, 1000, 2000, 4000)
    else:
        frontiers = (1500, 3000, 6000, 12000)
    budget = max(frontiers)
    policies = {}
    for offset, (label, policy, threshold) in enumerate(policy_specs):
        state = run_policy(
            prepared,
            targets,
            policy,
            budget,
            seed_value + 2003 * offset,
            hybrid_nullity_threshold=threshold,
        )
        policies[label] = compact_result(state.result(frontiers))

    preparation_work = work_total(prepared.preparation_work)
    primal_total = policies["rank-saturated"]["work_to_full_rank"]
    primary_total = policies["hybrid-d3"]["work_to_full_rank"]
    primal_online = None if primal_total is None else ZZ(primal_total) - preparation_work
    primary_online = None if primary_total is None else ZZ(primary_total) - preparation_work
    total_ratio = None if primal_total is None or primary_total is None else RR(primary_total) / RR(primal_total)
    online_ratio = None if primal_online is None or primary_online is None else RR(primary_online) / RR(primal_online)
    adaptive_total = policies["adaptive-dual"]["work_to_full_rank"]
    adaptive_online = None if adaptive_total is None else ZZ(adaptive_total) - preparation_work
    adaptive_total_ratio = None if primal_total is None or adaptive_total is None else RR(adaptive_total) / RR(primal_total)
    adaptive_online_ratio = None if primal_online is None or adaptive_online is None else RR(adaptive_online) / RR(primal_online)
    cell = {
        "fixture": fixture.as_dict(),
        "prepared": prepared.as_dict(),
        "policies": policies,
        "primary_hybrid_threshold": 3,
        "primary_total_work_ratio": None if total_ratio is None else float(total_ratio),
        "primary_online_work_ratio": None if online_ratio is None else float(online_ratio),
        "primal_online_work_to_full_rank": None if primal_online is None else int(primal_online),
        "primary_online_work_to_full_rank": None if primary_online is None else int(primary_online),
        "exploratory_adaptive_total_work_ratio": None if adaptive_total_ratio is None else float(adaptive_total_ratio),
        "exploratory_adaptive_online_work_ratio": None if adaptive_online_ratio is None else float(adaptive_online_ratio),
    }
    cells.append(cell)
    print(
        "%s width=%s online hybrid/primal=%s total=%s"
        % (fixture.label, prepared.width, cell["primary_online_work_ratio"], cell["primary_total_work_ratio"])
    )

bands = {}
for width in (13, 25):
    band_cells = [cell for cell in cells if cell["prepared"]["matrix_width"] == width]
    online_ratios = [RR(cell["primary_online_work_ratio"]) for cell in band_cells]
    total_ratios = [RR(cell["primary_total_work_ratio"]) for cell in band_cells]
    adaptive_online_ratios = [RR(cell["exploratory_adaptive_online_work_ratio"]) for cell in band_cells]
    adaptive_total_ratios = [RR(cell["exploratory_adaptive_total_work_ratio"]) for cell in band_cells]
    coefficient_ratios = [
        RR(cell["policies"]["hybrid-d3"]["work"]["coefficient_op"])
        / RR(cell["policies"]["rank-saturated"]["work"]["coefficient_op"])
        for cell in band_cells
    ]
    bands[str(width)] = {
        "cell_count": len(band_cells),
        "online_ratio_min": float(min(online_ratios)),
        "online_ratio_median": float(exact_median(online_ratios)),
        "online_ratio_max": float(max(online_ratios)),
        "total_ratio_median": float(exact_median(total_ratios)),
        "total_ratio_max": float(max(total_ratios)),
        "adaptive_online_ratio_median": float(exact_median(adaptive_online_ratios)),
        "adaptive_online_ratio_max": float(max(adaptive_online_ratios)),
        "adaptive_total_ratio_median": float(exact_median(adaptive_total_ratios)),
        "adaptive_total_ratio_max": float(max(adaptive_total_ratios)),
        "coefficient_work_ratio_median": float(exact_median(coefficient_ratios)),
        "coefficient_work_ratio_max": float(max(coefficient_ratios)),
    }

all_full = all(
    cell["policies"][label]["work_to_full_rank"] is not None
    for cell in cells
    for label in ("rank-saturated", "hybrid-d3")
)
verification_failures = sum(
    cell["policies"][label]["verification_failures"]
    for cell in cells
    for label, _, _ in policy_specs
)
passed = (
    all_full
    and verification_failures == 0
    and all(RR(band["online_ratio_median"]) < RR("0.90") for band in bands.values())
    and all(RR(cell["primary_online_work_ratio"]) <= 1 for cell in cells)
    and all(RR(band["total_ratio_max"]) <= 1 for band in bands.values())
)

summary = {
    "experiment": "SageMath exact dual-syndrome MCOLLAPSE filter",
    "sage_version": str(SAGE_VERSION),
    "held_out": True,
    "primary_hybrid_nullity_threshold": 3,
    "promotion_gate": {
        "passed": bool(passed),
        "verdict": "PROMOTE" if passed else "DO_NOT_PROMOTE",
    },
    "all_primary_policies_full_rank": bool(all_full),
    "verification_failures": int(verification_failures),
    "aggregate": {
        "primary_online_winning_cells": int(sum(RR(cell["primary_online_work_ratio"]) < 1 for cell in cells)),
        "adaptive_online_winning_cells": int(sum(RR(cell["exploratory_adaptive_online_work_ratio"]) < 1 for cell in cells)),
        "pure_dual_full_rank_cells": int(sum(cell["policies"]["dual-saturated"]["work_to_full_rank"] is not None for cell in cells)),
    },
    "bands": bands,
    "cells": cells,
}

output_directory = "mcollapse_sage/results"
os.makedirs(output_directory, exist_ok=True)
output_path = os.path.join(output_directory, "dual_filter_summary.json")
with open(output_path, "w", encoding="utf-8") as output_file:
    json.dump(json_safe(summary), output_file, indent=2, sort_keys=True)
    output_file.write("\n")

print("wrote %s" % output_path)
print("promotion gate: %s" % summary["promotion_gate"]["verdict"])
