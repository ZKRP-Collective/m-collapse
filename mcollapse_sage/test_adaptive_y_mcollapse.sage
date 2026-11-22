"""Tests for the frozen public-density y/point router."""

load("mcollapse_sage/rank_guided_mcollapse.sage")
load("mcollapse_sage/symmetry_quotient_mcollapse.sage")
load("mcollapse_sage/y_quotient_mcollapse.sage")


small = find_cm_fixture(8, 101)
large = find_cm_fixture(13, 103)
small_prepared = PreparedAdaptiveYPointMCollapse(small, 12)
large_prepared = PreparedAdaptiveYPointMCollapse(large, 4)
if small_prepared.use_y_quotient:
    raise AssertionError("dense small group should retain point lookup")
if not large_prepared.use_y_quotient:
    raise AssertionError("sparse large group should select y lookup")
for fixture, prepared in ((small, small_prepared), (large, large_prepared)):
    reference = PreparedQueryOrientedSymmetryQuotientMCollapse(
        fixture, prepared.base_size
    )
    for scalar in range(fixture.n):
        point = scalar * fixture.G
        if exact_pair_row_sets(reference, point) != exact_pair_row_sets(prepared, point):
            raise AssertionError("adaptive lookup changed an exact row set")
print("adaptive y/point MCOLLAPSE tests: PASS")

