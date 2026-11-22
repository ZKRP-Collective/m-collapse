import json
import os
from sage.env import SAGE_VERSION

load("mcollapse_sage/rank_guided_mcollapse.sage")
load("mcollapse_sage/symmetry_quotient_mcollapse.sage")


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
    (7, 211),
    (7, 223),
    (8, 227),
    (8, 229),
    (9, 233),
    (9, 239),
    (10, 241),
    (10, 251),
)
base_size = 12
target_count = 64
variable_budget = 3000
frontiers = (500, 1000, 2000, 3000)
cells = []

for subgroup_bits, seed_value in held_out:
    fixture = find_cm_fixture(subgroup_bits, seed_value)
    reference = PreparedWeightedMCollapse(fixture, base_size)
    quotient = PreparedSymmetryQuotientMCollapse(fixture, base_size)
    pair_audit = audit_pair_lookup_equivalence(reference, quotient)
    commutative_rows = unordered_pair_row_count(reference)
    _Q, targets = make_public_targets(fixture, target_count, seed_value + 2701)

    reference_state = run_policy(
        reference,
        targets,
        "rank-saturated",
        variable_budget,
        seed_value + 4001,
    )
    quotient_state = run_policy(
        quotient,
        targets,
        "rank-saturated",
        variable_budget,
        seed_value + 4001,
    )
    reference_result = compact_result(reference_state.result(frontiers))
    quotient_result = compact_result(quotient_state.result(frontiers))
    reference_full = reference_result["work_to_full_rank"]
    quotient_full = quotient_result["work_to_full_rank"]
    total_ratio = (
        None
        if reference_full is None or quotient_full is None
        else RR(quotient_full) / RR(reference_full)
    )
    reference_preparation = work_total(reference.preparation_work)
    quotient_preparation = work_total(quotient.preparation_work)
    preparation_ratio = RR(quotient_preparation) / RR(reference_preparation)
    reference_online = (
        None if reference_full is None else ZZ(reference_full) - reference_preparation
    )
    quotient_online = (
        None if quotient_full is None else ZZ(quotient_full) - quotient_preparation
    )
    online_ratio = (
        None
        if reference_online is None or quotient_online is None
        else RR(quotient_online) / RR(reference_online)
    )
    cell = {
        "fixture": fixture.as_dict(),
        "pair_audit": pair_audit,
        "commutative_exact_pair_rows": int(commutative_rows),
        "reference_prepared": reference.as_dict(),
        "quotient_prepared": quotient.as_dict(),
        "reference_policy": reference_result,
        "quotient_policy": quotient_result,
        "preparation_work_ratio": float(preparation_ratio),
        "reference_online_work_to_full_rank": None
        if reference_online is None
        else int(reference_online),
        "quotient_online_work_to_full_rank": None
        if quotient_online is None
        else int(quotient_online),
        "online_work_to_full_rank_ratio": None
        if online_ratio is None
        else float(online_ratio),
        "total_work_to_full_rank_ratio": None
        if total_ratio is None
        else float(total_ratio),
    }
    cells.append(cell)
    print(
        "%s rows-ok=%s reps=%s prep-ratio=%s full-ratio=%s"
        % (
            fixture.label,
            pair_audit["row_set_mismatches"] == 0,
            quotient.quotient_index.representative_count,
            float(preparation_ratio),
            cell["total_work_to_full_rank_ratio"],
        )
    )

ratios = [
    RR(cell["total_work_to_full_rank_ratio"])
    for cell in cells
    if cell["total_work_to_full_rank_ratio"] is not None
]
online_ratios = [
    RR(cell["online_work_to_full_rank_ratio"])
    for cell in cells
    if cell["online_work_to_full_rank_ratio"] is not None
]
all_pair_sets_equal = all(
    cell["pair_audit"]["row_set_mismatches"] == 0
    and cell["pair_audit"]["group_failures"] == 0
    for cell in cells
)
formula_matches = all(
    cell["quotient_prepared"]["quotient_pair_representatives"]
    == cell["quotient_prepared"]["predicted_pair_representatives"]
    for cell in cells
)
all_full_rank = all(
    cell["reference_policy"]["work_to_full_rank"] is not None
    and cell["quotient_policy"]["work_to_full_rank"] is not None
    for cell in cells
)
verification_failures = sum(
    cell[policy]["verification_failures"]
    for cell in cells
    for policy in ("reference_policy", "quotient_policy")
)
beats_commutative_control = all(
    cell["quotient_prepared"]["quotient_pair_representatives"]
    < cell["commutative_exact_pair_rows"]
    for cell in cells
)
passed = (
    len(ratios) == len(cells)
    and all_pair_sets_equal
    and formula_matches
    and all_full_rank
    and verification_failures == 0
    and exact_median(ratios) < RR("0.50")
    and max(ratios) < RR("0.70")
    and beats_commutative_control
)

summary = {
    "experiment": "SageMath exact S2 x mu6 weighted-MCOLLAPSE pair quotient",
    "sage_version": str(SAGE_VERSION),
    "held_out": True,
    "base_size": base_size,
    "target_count": target_count,
    "variable_work_budget": variable_budget,
    "aggregate": {
        "all_pair_row_sets_equal": bool(all_pair_sets_equal),
        "formula_matches_all_cells": bool(formula_matches),
        "all_policies_full_rank": bool(all_full_rank),
        "verification_failures": int(verification_failures),
        "beats_commutative_control": bool(beats_commutative_control),
        "total_ratio_min": float(min(ratios)),
        "total_ratio_median": float(exact_median(ratios)),
        "total_ratio_max": float(max(ratios)),
        "online_ratio_min": float(min(online_ratios)),
        "online_ratio_median": float(exact_median(online_ratios)),
        "online_ratio_max": float(max(online_ratios)),
        "winning_cells": int(sum(ratio < 1 for ratio in ratios)),
    },
    "promotion_gate": {
        "passed": bool(passed),
        "verdict": "PROMOTE" if passed else "DO_NOT_PROMOTE",
    },
    "cells": cells,
}

output_directory = "mcollapse_sage/results"
os.makedirs(output_directory, exist_ok=True)
output_path = os.path.join(output_directory, "symmetry_quotient_summary.json")
with open(output_path, "w", encoding="utf-8") as output_file:
    json.dump(json_safe(summary), output_file, indent=2, sort_keys=True)
    output_file.write("\n")

print("wrote %s" % output_path)
print("promotion gate: %s" % summary["promotion_gate"]["verdict"])
