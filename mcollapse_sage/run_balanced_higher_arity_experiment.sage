import hashlib
import json
import os
from sage.env import SAGE_VERSION

load("mcollapse_sage/rank_guided_mcollapse.sage")
load("mcollapse_sage/symmetry_quotient_mcollapse.sage")
load("mcollapse_sage/higher_arity_quotient.sage")
load("mcollapse_sage/balanced_higher_arity.sage")


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


def relation_digest(result):
    digest = hashlib.sha256()
    for relation in result.relations:
        digest.update(repr(relation.row_key).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def prefix_rank(prepared, target, result):
    basis = OnlineSparseBasis(prepared.fixture.K, prepared.width)
    coefficient_work = ZZ(0)
    relations_to_cap = None
    for relation_index, relation in enumerate(result.relations, start=1):
        row = {
            ZZ(anchor): prepared.fixture.K(coefficient)
            for anchor, coefficient in relation.row_key
        }
        row[prepared.width - 1] = -prepared.fixture.K(target.v)
        residual, operations = basis.reduce(row)
        coefficient_work += operations
        if residual:
            basis.commit(residual)
        if basis.rank == len(prepared.representatives):
            relations_to_cap = relation_index
            break
    return int(basis.rank), int(coefficient_work), relations_to_cap


def clone_basis(basis):
    cloned = OnlineSparseBasis(basis.field, basis.width)
    cloned.pivots = {
        pivot: dict(row) for pivot, row in basis.pivots.items()
    }
    return cloned


def apply_relations(prepared, basis, target, relations, stop_rank):
    coefficient_work = ZZ(0)
    used = ZZ(0)
    for relation in relations:
        row = {
            ZZ(anchor): prepared.fixture.K(coefficient)
            for anchor, coefficient in relation.row_key
        }
        row[prepared.width - 1] = -prepared.fixture.K(target.v)
        residual, operations = basis.reduce(row)
        coefficient_work += operations
        if residual:
            basis.commit(residual)
        used += 1
        if basis.rank >= stop_rank:
            break
    return used, coefficient_work


def solve_to_full_rank(prepared, generator, targets, relation_arity):
    basis = OnlineSparseBasis(prepared.fixture.K, prepared.width)
    generation_work = ZZ(0)
    coefficient_work = ZZ(0)
    relations_used = ZZ(0)
    complement_lookups = ZZ(0)
    processed_targets = ZZ(0)
    for target in targets:
        if basis.rank == prepared.width:
            break
        processed_targets += 1
        stop_rank = min(
            prepared.width,
            basis.rank + len(prepared.representatives),
        )
        trial = clone_basis(basis)
        probe = generator.generate(
            target, relation_arity, max_unique=4 * base_size
        )
        needed, _trial_work = apply_relations(
            prepared, trial, target, probe.relations, stop_rank
        )
        if trial.rank < stop_rank:
            raise ArithmeticError("balanced prefix did not reach its exact rank cap")
        exact = generator.generate(
            target, relation_arity, max_unique=needed
        )
        used, reduction_work = apply_relations(
            prepared, basis, target, exact.relations, stop_rank
        )
        if used != needed:
            raise ArithmeticError("balanced exact stop is not deterministic")
        generation_work += work_total(exact.work)
        coefficient_work += reduction_work
        relations_used += used
        complement_lookups += exact.stats["complement_lookups"]
    return {
        "rank": int(basis.rank),
        "matrix_width": int(prepared.width),
        "processed_targets": int(processed_targets),
        "relations_used": int(relations_used),
        "complement_lookups": int(complement_lookups),
        "generation_work": int(generation_work),
        "rank_coefficient_work": int(coefficient_work),
        "total_with_pair_preparation": int(
            work_total(prepared.preparation_work)
            + generation_work
            + coefficient_work
        ),
    }


held_out = ((7, 401), (8, 409), (9, 419), (10, 421), (11, 431), (12, 433))
base_size = 4
arity_four_control = HigherArityCoefficientQuotient(3).global_count(base_size)
arity_five_control = HigherArityCoefficientQuotient(4).global_count(base_size)
cells = []

for subgroup_bits, seed_value in held_out:
    fixture = find_cm_fixture(subgroup_bits, seed_value)
    prepared = PreparedSymmetryQuotientMCollapse(fixture, base_size)
    generator = BalancedHigherArityGenerator(prepared)
    _Q, targets = make_public_targets(fixture, 4, seed_value + 3701)
    target = targets[0]
    arity_three_state = run_policy(
        prepared,
        targets,
        "rank-saturated",
        10000,
        seed_value + 4703,
    )
    arity_three_result = arity_three_state.result((10000,))
    arity_three_total = arity_three_result["work_to_full_rank"]
    if arity_three_total is None:
        raise ArithmeticError("arity-three reference did not reach full rank")
    preparation_work = work_total(prepared.preparation_work)
    arity_three_online = ZZ(arity_three_total) - preparation_work
    results = {}
    for relation_arity in (4, 5):
        generated = generator.generate(target, relation_arity)
        item = generated.as_dict()
        item["row_sha256"] = relation_digest(generated)
        item["total_with_pair_preparation"] = int(
            work_total(prepared.preparation_work) + work_total(generated.work)
        )
        prefix = generator.generate(
            target, relation_arity, max_unique=4 * base_size
        )
        rank, rank_work, relations_to_cap = prefix_rank(
            prepared, target, prefix
        )
        if relations_to_cap is not None and relations_to_cap < len(prefix.relations):
            prefix = generator.generate(
                target, relation_arity, max_unique=relations_to_cap
            )
            rank, rank_work, repeated_cap = prefix_rank(
                prepared, target, prefix
            )
            if repeated_cap != relations_to_cap:
                raise ArithmeticError("rank saturation prefix is not deterministic")
        item["saturated_prefix"] = {
            "relation_limit": int(4 * base_size),
            "relations_generated": int(len(prefix.relations)),
            "relations_to_rank_cap": None
            if relations_to_cap is None
            else int(relations_to_cap),
            "rank": rank,
            "rank_cap": base_size,
            "generation_work": int(work_total(prefix.work)),
            "rank_coefficient_work": rank_work,
            "total_with_pair_preparation": int(
                work_total(prepared.preparation_work)
                + work_total(prefix.work)
                + rank_work
            ),
            "complement_lookups": int(prefix.stats["complement_lookups"]),
        }
        item["solver_to_full_rank"] = solve_to_full_rank(
            prepared, generator, targets, relation_arity
        )
        solver = item["solver_to_full_rank"]
        solver_online = ZZ(solver["generation_work"]) + ZZ(
            solver["rank_coefficient_work"]
        )
        solver["online_work"] = int(solver_online)
        solver["total_vs_arity_3_ratio"] = float(
            RR(solver["total_with_pair_preparation"])
            / RR(arity_three_total)
        )
        solver["online_vs_arity_3_ratio"] = float(
            RR(solver_online) / RR(arity_three_online)
        )
        results[str(relation_arity)] = item
    expected_exact_pairs = 6 * (
        prepared.quotient_index.representative_count - 1
    ) + 1
    cell = {
        "fixture": fixture.as_dict(),
        "prepared": prepared.as_dict(),
        "target": {
            "index": int(target.index),
            "u": int(target.u),
            "v": int(target.v),
            "point": [int(target.point[0]), int(target.point[1])],
        },
        "stored_pair_representatives": int(
            prepared.quotient_index.representative_count
        ),
        "streamed_exact_pair_rows": int(generator.exact_pair_count),
        "expected_exact_pair_rows": int(expected_exact_pairs),
        "full_partial_controls": {
            "arity-4": int(arity_four_control),
            "arity-5": int(arity_five_control),
        },
        "results": results,
        "arity_3_reference": {
            "rank": int(arity_three_state.basis.rank),
            "matrix_width": int(prepared.width),
            "processed_targets": int(arity_three_state.processed_targets),
            "total_work_to_full_rank": int(arity_three_total),
            "online_work_to_full_rank": int(arity_three_online),
        },
    }
    cells.append(cell)
    print(
        "%s arity4=%s arity5=%s lookups5=%s failures=%s/%s"
        % (
            fixture.label,
            results["4"]["relation_count"],
            results["5"]["relation_count"],
            results["5"]["stats"]["complement_lookups"],
            results["4"]["stats"]["group_failures"],
            results["5"]["stats"]["group_failures"],
        )
    )

all_phase_counts = all(
    cell["streamed_exact_pair_rows"] == cell["expected_exact_pair_rows"]
    for cell in cells
)
all_nonempty = all(
    cell["results"][str(arity)]["relation_count"] > 0
    for cell in cells
    for arity in (4, 5)
)
group_failures = sum(
    cell["results"][str(arity)]["stats"]["group_failures"]
    for cell in cells
    for arity in (4, 5)
)
all_prefixes_saturate = all(
    cell["results"][str(arity)]["saturated_prefix"]["rank"] == base_size
    for cell in cells
    for arity in (4, 5)
)
all_solvers_full_rank = all(
    cell["results"][str(arity)]["solver_to_full_rank"]["rank"]
    == cell["results"][str(arity)]["solver_to_full_rank"]["matrix_width"]
    for cell in cells
    for arity in (4, 5)
)
storage_beats_controls = all(
    cell["stored_pair_representatives"]
    < cell["full_partial_controls"][label]
    for cell in cells
    for label in ("arity-4", "arity-5")
)
passed = (
    all_phase_counts
    and all_nonempty
    and group_failures == 0
    and storage_beats_controls
    and all_prefixes_saturate
    and all_solvers_full_rank
)

arity_four_total_ratios = [
    RR(cell["results"]["4"]["solver_to_full_rank"]["total_vs_arity_3_ratio"])
    for cell in cells
]
arity_five_total_ratios = [
    RR(cell["results"]["5"]["solver_to_full_rank"]["total_vs_arity_3_ratio"])
    for cell in cells
]
arity_four_online_ratios = [
    RR(cell["results"]["4"]["solver_to_full_rank"]["online_vs_arity_3_ratio"])
    for cell in cells
]
arity_five_online_ratios = [
    RR(cell["results"]["5"]["solver_to_full_rank"]["online_vs_arity_3_ratio"])
    for cell in cells
]

summary = {
    "experiment": "SageMath balanced higher-arity MCOLLAPSE generator",
    "sage_version": str(SAGE_VERSION),
    "held_out": True,
    "base_size": base_size,
    "aggregate": {
        "all_phase_counts_exact": bool(all_phase_counts),
        "all_cells_nonempty": bool(all_nonempty),
        "group_failures": int(group_failures),
        "storage_beats_full_partial_controls": bool(storage_beats_controls),
        "all_prefixes_reach_target_rank_cap": bool(all_prefixes_saturate),
        "all_balanced_solvers_full_rank": bool(all_solvers_full_rank),
        "arity_4_relations_total": int(
            sum(cell["results"]["4"]["relation_count"] for cell in cells)
        ),
        "arity_5_relations_total": int(
            sum(cell["results"]["5"]["relation_count"] for cell in cells)
        ),
        "exploratory_arity_4_total_ratio_median": float(
            exact_median(arity_four_total_ratios)
        ),
        "exploratory_arity_4_total_ratio_max": float(
            max(arity_four_total_ratios)
        ),
        "exploratory_arity_5_total_ratio_median": float(
            exact_median(arity_five_total_ratios)
        ),
        "exploratory_arity_5_total_ratio_max": float(
            max(arity_five_total_ratios)
        ),
        "exploratory_arity_4_online_ratio_median": float(
            exact_median(arity_four_online_ratios)
        ),
        "exploratory_arity_5_online_ratio_median": float(
            exact_median(arity_five_online_ratios)
        ),
    },
    "promotion_gate": {
        "passed": bool(passed),
        "verdict": "PROMOTE_BALANCED_STORAGE" if passed else "DO_NOT_PROMOTE",
    },
    "cells": cells,
}

output_directory = "mcollapse_sage/results"
os.makedirs(output_directory, exist_ok=True)
output_path = os.path.join(output_directory, "balanced_higher_arity_summary.json")
with open(output_path, "w", encoding="utf-8") as output_file:
    json.dump(json_safe(summary), output_file, indent=2, sort_keys=True)
    output_file.write("\n")

print("wrote %s" % output_path)
print("promotion gate: %s" % summary["promotion_gate"]["verdict"])
