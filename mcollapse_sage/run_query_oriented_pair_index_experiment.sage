import hashlib
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


def trace_digest(trace):
    digest = hashlib.sha256()
    for item in trace:
        digest.update(repr(item).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def exhaustive_index_comparison(canonical, query_oriented):
    fixture = canonical.fixture
    mismatches = ZZ(0)
    group_failures = ZZ(0)
    exact_rows = ZZ(0)
    for scalar in range(fixture.n):
        point = scalar * fixture.G
        canonical_pairs, _canonical_work = canonical.lookup_pairs(point)
        query_pairs, _query_work = query_oriented.lookup_pairs(point)
        canonical_rows = {
            sparse_pair_row_key(fixture, left, right)
            for left, right in canonical_pairs
        }
        query_rows = {
            sparse_pair_row_key(fixture, left, right)
            for left, right in query_pairs
        }
        exact_rows += len(canonical_rows)
        if canonical_rows != query_rows:
            mismatches += 1
        group_failures += sum(
            left.point + right.point != point for left, right in query_pairs
        )
    return {
        "subgroup_points_checked": int(fixture.n),
        "exact_rows_checked": int(exact_rows),
        "row_set_mismatches": int(mismatches),
        "group_failures": int(group_failures),
    }


held_out = (
    (8, 809),
    (9, 811),
    (10, 821),
    (11, 823),
    (12, 827),
    (13, 829),
)
base_size = 12
target_count = 4
variable_budget = 250000
cells = []

for subgroup_bits, seed_value in held_out:
    fixture = find_cm_fixture(subgroup_bits, seed_value)
    canonical = PreparedSymmetryQuotientMCollapse(fixture, base_size)
    query_oriented = PreparedQueryOrientedSymmetryQuotientMCollapse(
        fixture, base_size
    )
    audit = exhaustive_index_comparison(canonical, query_oriented)
    _Q, targets = make_public_targets(fixture, target_count, seed_value + 12101)

    canonical_state = run_policy(
        canonical,
        targets,
        "rank-saturated",
        variable_budget,
        seed_value + 13103,
        window_size=1,
    )
    query_state = run_policy(
        query_oriented,
        targets,
        "rank-saturated",
        variable_budget,
        seed_value + 13103,
        window_size=1,
    )
    canonical_result = canonical_state.result((variable_budget,))
    query_result = query_state.result((variable_budget,))
    canonical_full = canonical_result["work_to_full_rank"]
    query_full = query_result["work_to_full_rank"]
    if canonical_full is None or query_full is None:
        raise ArithmeticError("pair-index solver did not reach full rank")
    canonical_preparation = work_total(canonical.preparation_work)
    query_preparation = work_total(query_oriented.preparation_work)
    candidate_trace_equal = canonical_state.candidate_trace == query_state.candidate_trace
    pivot_trace_equal = canonical_state.pivot_trace == query_state.pivot_trace
    final_basis_equal = canonical_state.basis.dense_rows() == query_state.basis.dense_rows()

    cell = {
        "fixture": fixture.as_dict(),
        "audit": audit,
        "canonical_prepared": canonical.as_dict(),
        "query_oriented_prepared": query_oriented.as_dict(),
        "stored_representatives_equal": bool(
            canonical.quotient_index.representative_count
            == query_oriented.quotient_index.representative_count
        ),
        "candidate_trace_equal": bool(candidate_trace_equal),
        "pivot_trace_equal": bool(pivot_trace_equal),
        "final_basis_equal": bool(final_basis_equal),
        "candidate_trace_sha256": trace_digest(query_state.candidate_trace),
        "pivot_trace_sha256": trace_digest(query_state.pivot_trace),
        "canonical": canonical_result,
        "query_oriented": query_result,
        "preparation_work_ratio": float(
            RR(query_preparation) / RR(canonical_preparation)
        ),
        "online_work_ratio": float(
            RR(ZZ(query_full) - query_preparation)
            / RR(ZZ(canonical_full) - canonical_preparation)
        ),
        "total_work_ratio": float(RR(query_full) / RR(canonical_full)),
    }
    cells.append(cell)
    print(
        "%s prep=%s total=%s mismatches=%s traces=%s/%s"
        % (
            fixture.label,
            cell["preparation_work_ratio"],
            cell["total_work_ratio"],
            audit["row_set_mismatches"],
            candidate_trace_equal,
            pivot_trace_equal,
        )
    )

preparation_ratios = [RR(cell["preparation_work_ratio"]) for cell in cells]
online_ratios = [RR(cell["online_work_ratio"]) for cell in cells]
total_ratios = [RR(cell["total_work_ratio"]) for cell in cells]
all_lookups_equal = all(
    cell["audit"]["row_set_mismatches"] == 0
    and cell["audit"]["group_failures"] == 0
    for cell in cells
)
all_traces_equal = all(
    cell["candidate_trace_equal"]
    and cell["pivot_trace_equal"]
    and cell["final_basis_equal"]
    for cell in cells
)
all_counts_equal = all(cell["stored_representatives_equal"] for cell in cells)
all_full_rank = all(
    cell[mode]["rank"] == cell[mode]["matrix_width"]
    for cell in cells
    for mode in ("canonical", "query_oriented")
)
verification_failures = sum(
    cell[mode]["verification_failures"]
    for cell in cells
    for mode in ("canonical", "query_oriented")
)
median_preparation = exact_median(preparation_ratios)
median_total = exact_median(total_ratios)
worst_total = max(total_ratios)
passed = (
    all_lookups_equal
    and all_traces_equal
    and all_counts_equal
    and all_full_rank
    and verification_failures == 0
    and median_preparation <= RR("0.65")
    and median_total <= RR("0.75")
    and worst_total < RR("0.80")
)

summary = {
    "experiment": "SageMath query-oriented exact MCOLLAPSE pair index",
    "sage_version": str(SAGE_VERSION),
    "preregistered": True,
    "base_size": base_size,
    "matrix_width": base_size + 1,
    "target_count": target_count,
    "held_out_cells": len(held_out),
    "aggregate": {
        "all_exhaustive_lookups_equal": bool(all_lookups_equal),
        "all_candidate_and_pivot_traces_equal": bool(all_traces_equal),
        "all_stored_counts_equal": bool(all_counts_equal),
        "all_solvers_full_rank": bool(all_full_rank),
        "verification_failures": int(verification_failures),
        "median_preparation_work_ratio": float(median_preparation),
        "median_online_work_ratio": float(exact_median(online_ratios)),
        "median_total_work_ratio": float(median_total),
        "worst_total_work_ratio": float(worst_total),
    },
    "promotion_gate": {
        "passed": bool(passed),
        "verdict": "PROMOTE_QUERY_ORIENTED_INDEX" if passed else "KEEP_CANONICAL_INDEX",
    },
    "cells": cells,
}

output_directory = "mcollapse_sage/results"
os.makedirs(output_directory, exist_ok=True)
output_path = os.path.join(
    output_directory, "query_oriented_pair_index_summary.json"
)
with open(output_path, "w", encoding="utf-8") as output_file:
    json.dump(json_safe(summary), output_file, indent=2, sort_keys=True)
    output_file.write("\n")

with open(output_path, "rb") as input_file:
    output_sha256 = hashlib.sha256(input_file.read()).hexdigest()

print("wrote %s" % output_path)
print("summary sha256: %s" % output_sha256)
print("promotion gate: %s" % summary["promotion_gate"]["verdict"])
