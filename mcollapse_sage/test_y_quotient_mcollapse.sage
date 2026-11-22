"""Exact tests for the GLV-invariant y-fiber pair index."""

load("mcollapse_sage/rank_guided_mcollapse.sage")
load("mcollapse_sage/symmetry_quotient_mcollapse.sage")
load("mcollapse_sage/y_quotient_mcollapse.sage")


for subgroup_bits, seed_value in ((7, 73), (8, 79), (9, 83)):
    fixture = find_cm_fixture(subgroup_bits, seed_value)
    reference = PreparedQueryOrientedSymmetryQuotientMCollapse(fixture, 4)
    candidate = PreparedYQuotientMCollapse(fixture, 4)
    audit = audit_y_lookup_equivalence(reference, candidate)
    if audit["row_set_mismatches"] or audit["group_failures"]:
        raise AssertionError("y quotient changed an exact pair lookup")
print("Y-quotient MCOLLAPSE tests: PASS")

