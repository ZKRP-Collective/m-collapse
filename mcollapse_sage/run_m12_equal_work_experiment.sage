import hashlib
import json
import os
from sage.env import SAGE_VERSION

load("mcollapse_sage/rank_guided_mcollapse.sage")
load("mcollapse_sage/symmetry_quotient_mcollapse.sage")
load("mcollapse_sage/balanced_higher_arity.sage")
load("mcollapse_sage/equal_work_mcollapse.sage")


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


held_out = (
    (8, 701),
    (9, 709),
    (10, 719),
    (11, 727),
    (12, 733),
    (13, 743),
)
base_size = 12
target_count = 4
variable_budget = 250000
cells = []

for subgroup_bits, seed_value in held_out:
    fixture = find_cm_fixture(subgroup_bits, seed_value)
    prepared = PreparedSymmetryQuotientMCollapse(fixture, base_size)
    _Q, targets = make_public_targets(fixture, target_count, seed_value + 8101)
    preparation_work = work_total(prepared.preparation_work)

    reference_state = run_policy(
        prepared,
        targets,
        "rank-saturated",
        variable_budget,
        seed_value + 9103,
        window_size=1,
    )
    reference_probe = reference_state.result((variable_budget,))
    reference_full_work = reference_probe["work_to_full_rank"]
    if reference_full_work is None:
        raise ArithmeticError("arity-three reference did not reach full rank")
    reference_variable_frontier = ZZ(reference_full_work) - preparation_work
    reference = reference_state.result((reference_variable_frontier,))

    policies = {"arity-3": reference}
    for relation_arity in (4, 5):
        balanced_run = run_balanced_equal_work(
            prepared,
            targets,
            relation_arity,
            variable_budget,
            seed_value + 10007 * relation_arity,
        )
        result = balanced_run.result((reference_variable_frontier,))
        full_work = result["work_to_full_rank"]
        if full_work is None:
            ratio = None
        else:
            ratio = RR(full_work) / RR(reference_full_work)
        result["rank_at_arity_3_full_work"] = int(
            balanced_run.state.rank_at(reference_variable_frontier)
        )
        result["total_work_ratio_to_arity_3"] = (
            None if ratio is None else float(ratio)
        )
        result["online_work_to_full_rank"] = (
            None if full_work is None else int(ZZ(full_work) - preparation_work)
        )
        result["online_work_ratio_to_arity_3"] = (
            None
            if full_work is None
            else float(
                RR(ZZ(full_work) - preparation_work)
                / RR(ZZ(reference_full_work) - preparation_work)
            )
        )
        policies["arity-%s" % relation_arity] = result

    cell = {
        "fixture": fixture.as_dict(),
        "prepared": prepared.as_dict(),
        "target_count": target_count,
        "variable_work_ceiling": variable_budget,
        "equal_work_total_frontier": int(reference_full_work),
        "equal_work_variable_frontier": int(reference_variable_frontier),
        "policies": policies,
    }
    cells.append(cell)
    print(
        "%s ref=%s a4=%s (%s) a5=%s (%s)"
        % (
            fixture.label,
            reference_full_work,
            policies["arity-4"]["work_to_full_rank"],
            policies["arity-4"]["total_work_ratio_to_arity_3"],
            policies["arity-5"]["work_to_full_rank"],
            policies["arity-5"]["total_work_ratio_to_arity_3"],
        )
    )

all_full_rank = all(
    cell["policies"][label]["rank"]
    == cell["policies"][label]["matrix_width"]
    for cell in cells
    for label in ("arity-3", "arity-4", "arity-5")
)
verification_failures = sum(
    cell["policies"][label]["verification_failures"]
    for cell in cells
    for label in ("arity-3", "arity-4", "arity-5")
)

promotion = {}
for label in ("arity-4", "arity-5"):
    ratios = [
        RR(cell["policies"][label]["total_work_ratio_to_arity_3"])
        for cell in cells
    ]
    equal_work_full_rank = sum(
        cell["policies"][label]["rank_at_arity_3_full_work"]
        == cell["policies"][label]["matrix_width"]
        for cell in cells
    )
    median_ratio = exact_median(ratios)
    worst_ratio = max(ratios)
    online_ratios = [
        RR(cell["policies"][label]["online_work_ratio_to_arity_3"])
        for cell in cells
    ]
    passed = (
        all_full_rank
        and verification_failures == 0
        and equal_work_full_rank >= 5
        and median_ratio <= RR("0.90")
        and worst_ratio < 1
    )
    promotion[label] = {
        "equal_work_full_rank_cells": int(equal_work_full_rank),
        "median_total_work_ratio": float(median_ratio),
        "worst_total_work_ratio": float(worst_ratio),
        "median_online_work_ratio": float(exact_median(online_ratios)),
        "worst_online_work_ratio": float(max(online_ratios)),
        "passed": bool(passed),
    }

promoted = [label for label in ("arity-4", "arity-5") if promotion[label]["passed"]]
summary = {
    "experiment": "SageMath m=12 equal-work balanced MCOLLAPSE",
    "sage_version": str(SAGE_VERSION),
    "preregistered": True,
    "base_size": base_size,
    "matrix_width": base_size + 1,
    "target_count": target_count,
    "variable_work_ceiling": variable_budget,
    "held_out_cells": len(held_out),
    "aggregate": {
        "all_policies_full_rank": bool(all_full_rank),
        "verification_failures": int(verification_failures),
        "median_reference_preparation_fraction": float(
            exact_median([
                RR(cell["prepared"]["preparation_primitive_work"])
                / RR(cell["policies"]["arity-3"]["work_to_full_rank"])
                for cell in cells
            ])
        ),
        "promotion": promotion,
    },
    "promotion_gate": {
        "passed": bool(promoted),
        "promoted_arities": promoted,
        "verdict": (
            "PROMOTE_%s" % "_AND_".join(label.upper().replace("-", "_") for label in promoted)
            if promoted
            else "KEEP_BALANCED_AS_STORAGE_ONLY"
        ),
    },
    "cells": cells,
}

output_directory = "mcollapse_sage/results"
os.makedirs(output_directory, exist_ok=True)
output_path = os.path.join(output_directory, "m12_equal_work_summary.json")
with open(output_path, "w", encoding="utf-8") as output_file:
    json.dump(json_safe(summary), output_file, indent=2, sort_keys=True)
    output_file.write("\n")

with open(output_path, "rb") as input_file:
    output_sha256 = hashlib.sha256(input_file.read()).hexdigest()

print("wrote %s" % output_path)
print("summary sha256: %s" % output_sha256)
print("promotion gate: %s" % summary["promotion_gate"]["verdict"])
