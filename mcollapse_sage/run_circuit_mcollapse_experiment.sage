import hashlib
import json
import os
from sage.env import SAGE_VERSION

load("mcollapse_sage/rank_guided_mcollapse.sage")
load("mcollapse_sage/symmetry_quotient_mcollapse.sage")
load("mcollapse_sage/circuit_mcollapse.sage")


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
    (8, 907),
    (9, 911),
    (10, 919),
    (11, 929),
    (12, 937),
    (13, 941),
)
base_size = 12
target_count = 32
variable_budget = 250000
cells = []
random_target_audit = os.environ.get("MCOLLAPSE_RANDOM_TARGET_AUDIT") == "1"
target_factory = (
    make_distinct_random_public_targets
    if random_target_audit
    else make_public_targets
)

for subgroup_bits, seed_value in held_out:
    fixture = find_cm_fixture(subgroup_bits, seed_value)
    prepared = PreparedQueryOrientedSymmetryQuotientMCollapse(fixture, base_size)
    Q, targets = target_factory(fixture, target_count, seed_value + 15101)
    distinct_target_points = len({point_key(target.point) for target in targets})
    preparation_work = work_total(prepared.preparation_work)

    reference_state = run_policy(
        prepared,
        targets,
        "rank-saturated",
        variable_budget,
        seed_value + 16103,
        window_size=1,
    )
    reference = reference_state.result((variable_budget,))
    reference_total = reference["work_to_full_rank"]
    if reference_total is None:
        raise ArithmeticError("augmented reference did not reach full rank")
    reference_online = ZZ(reference_total) - preparation_work

    circuit_run = run_circuit_mcollapse(
        prepared,
        targets,
        variable_budget,
        seed_value + 17107,
        round_robin=True,
    )
    circuit = circuit_run.as_dict()
    circuit_total = ZZ(circuit["total_with_preparation"])
    circuit_online = ZZ(circuit["variable_work"])
    recovered_exactly = bool(
        circuit_run.solved and ZZ(circuit_run.recovered) * fixture.G == Q
    )

    cell = {
        "fixture": fixture.as_dict(),
        "prepared": prepared.as_dict(),
        "target_count": target_count,
        "distinct_target_points": distinct_target_points,
        "variable_work_ceiling": variable_budget,
        "reference": reference,
        "circuit": circuit,
        "recovered_exactly": recovered_exactly,
        "reference_online_work": int(reference_online),
        "circuit_online_work": int(circuit_online),
        "online_work_ratio": float(RR(circuit_online) / RR(reference_online)),
        "total_work_ratio": float(RR(circuit_total) / RR(reference_total)),
    }
    cells.append(cell)
    print(
        "%s relations=%s anchor-rank=%s online=%s total=%s exact=%s"
        % (
            fixture.label,
            circuit["relations_processed"],
            circuit["anchor_rank"],
            cell["online_work_ratio"],
            cell["total_work_ratio"],
            recovered_exactly,
        )
    )

all_correct = all(
    cell["reference"]["rank"] == cell["reference"]["matrix_width"]
    and cell["reference"]["verification_failures"] == 0
    and cell["circuit"]["solved"]
    and cell["circuit"]["verification_failures"] == 0
    and cell["recovered_exactly"]
    and cell["circuit"]["exact_rank_audit"]["online_anchor_rank_matches"]
    and cell["circuit"]["exact_rank_audit"]["system_consistent"]
    and cell["circuit"]["exact_rank_audit"]["target_column_adds_rank"]
    and (
        not random_target_audit
        or cell["distinct_target_points"] == cell["target_count"]
    )
    for cell in cells
)
early_rank_cells = sum(cell["circuit"]["anchor_rank"] < base_size for cell in cells)
relations = [ZZ(cell["circuit"]["relations_processed"]) for cell in cells]
online_ratios = [RR(cell["online_work_ratio"]) for cell in cells]
total_ratios = [RR(cell["total_work_ratio"]) for cell in cells]
online_wins = sum(ratio < 1 for ratio in online_ratios)
median_relations = exact_median(relations)
median_online_ratio = exact_median(online_ratios)
median_total_ratio = exact_median(total_ratios)
passed = (
    all_correct
    and early_rank_cells >= 5
    and median_relations <= RR("0.85") * (base_size + 1)
    and median_online_ratio <= RR("0.75")
    and online_wins >= 5
    and median_total_ratio <= RR("0.97")
)

summary = {
    "experiment": (
        "SageMath Circuit-MCOLLAPSE distinct-random-target robustness audit"
        if random_target_audit
        else "SageMath preregistered Circuit-MCOLLAPSE stopping rule"
    ),
    "sage_version": str(SAGE_VERSION),
    "preregistered": not random_target_audit,
    "preregistration": "CIRCUIT_MCOLLAPSE_PREREGISTRATION.md",
    "target_generation": (
        "distinct_sage_random_public_combinations"
        if random_target_audit
        else "frozen_structured_public_combinations"
    ),
    "base_size": base_size,
    "target_count": target_count,
    "variable_work_ceiling": variable_budget,
    "held_out_cells": len(held_out),
    "aggregate": {
        "all_correctness_checks_passed": bool(all_correct),
        "early_anchor_rank_cells": int(early_rank_cells),
        "online_work_win_cells": int(online_wins),
        "median_relations_processed": float(median_relations),
        "median_online_work_ratio": float(median_online_ratio),
        "worst_online_work_ratio": float(max(online_ratios)),
        "median_total_work_ratio": float(median_total_ratio),
        "worst_total_work_ratio": float(max(total_ratios)),
    },
    "promotion_gate": {
        "passed": bool(passed),
        "verdict": (
            (
                "CONFIRM_CIRCUIT_MCOLLAPSE_ON_DISTINCT_RANDOM_TARGETS"
                if random_target_audit
                else "PROMOTE_CIRCUIT_MCOLLAPSE_STOPPING_RULE"
            )
            if passed
            else (
                "RANDOM_TARGET_AUDIT_FAILED"
                if random_target_audit
                else "REJECT_CIRCUIT_MCOLLAPSE_STOPPING_RULE"
            )
        ),
    },
    "cells": cells,
}

output_directory = "mcollapse_sage/results"
os.makedirs(output_directory, exist_ok=True)
output_path = os.path.join(
    output_directory,
    (
        "circuit_random_target_summary.json"
        if random_target_audit
        else "circuit_mcollapse_summary.json"
    ),
)
with open(output_path, "w", encoding="utf-8") as output_file:
    json.dump(json_safe(summary), output_file, indent=2, sort_keys=True)
    output_file.write("\n")

with open(output_path, "rb") as input_file:
    output_sha256 = hashlib.sha256(input_file.read()).hexdigest()

print("wrote %s" % output_path)
print("summary sha256: %s" % output_sha256)
print("promotion gate: %s" % summary["promotion_gate"]["verdict"])
