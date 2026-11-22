load("mcollapse_sage/rank_guided_mcollapse.sage")
load("mcollapse_sage/symmetry_quotient_mcollapse.sage")


def test_closed_count_and_exact_lookup_coverage():
    fixture = find_cm_fixture(7, 11)
    reference = PreparedWeightedMCollapse(fixture, 12)
    quotient = PreparedSymmetryQuotientMCollapse(fixture, 12)
    predicted = 24 * binomial(12, 2) + 8 * 12 + 1
    assert predicted == 1681
    assert quotient.quotient_index.representative_count == predicted
    assert unordered_pair_row_count(reference) == 10081
    audit = audit_pair_lookup_equivalence(reference, quotient)
    assert audit["subgroup_points_checked"] == fixture.n
    assert audit["exact_pair_rows_checked"] == 10081
    assert audit["row_set_mismatches"] == 0
    assert audit["group_failures"] == 0


def test_quotient_solver_reaches_the_same_exact_rank():
    fixture = find_cm_fixture(7, 19)
    reference = PreparedWeightedMCollapse(fixture, 12)
    quotient = PreparedSymmetryQuotientMCollapse(fixture, 12)
    _Q, targets = make_public_targets(fixture, 64, 1920)
    reference_state = run_policy(
        reference, targets, "rank-saturated", 3000, 73
    )
    quotient_state = run_policy(
        quotient, targets, "rank-saturated", 3000, 73
    )
    assert reference_state.verification_failures == 0
    assert quotient_state.verification_failures == 0
    assert reference_state.basis.rank == reference.width
    assert quotient_state.basis.rank == quotient.width
    assert reference_state.basis.rank == quotient_state.basis.rank
    assert matrix(
        fixture.K, reference_state.basis.dense_rows()
    ).rank() == reference.width
    assert matrix(
        fixture.K, quotient_state.basis.dense_rows()
    ).rank() == quotient.width
    assert work_total(quotient.preparation_work) < work_total(
        reference.preparation_work
    )


def test_invalid_window_remains_rejected_through_generic_lookup():
    fixture = find_cm_fixture(7, 11)
    quotient = PreparedSymmetryQuotientMCollapse(fixture, 12)
    _Q, targets = make_public_targets(fixture, 2, 17)
    try:
        run_policy(
            quotient,
            targets,
            "rank-saturated",
            100,
            19,
            window_size=0,
        )
        raise AssertionError("zero window was accepted")
    except ValueError:
        pass


def test_query_oriented_index_matches_every_exact_lookup():
    fixture = find_cm_fixture(8, 29)
    canonical = PreparedSymmetryQuotientMCollapse(fixture, 12)
    query_oriented = PreparedQueryOrientedSymmetryQuotientMCollapse(fixture, 12)
    assert (
        canonical.quotient_index.representative_count
        == query_oriented.quotient_index.representative_count
        == 1681
    )
    assert work_total(query_oriented.preparation_work) < work_total(
        canonical.preparation_work
    )
    assert query_oriented.quotient_index.work["endomorphism"] == 0
    for scalar in range(fixture.n):
        point = scalar * fixture.G
        canonical_rows = exact_pair_row_sets(canonical, point)
        query_rows = exact_pair_row_sets(query_oriented, point)
        assert canonical_rows == query_rows
        pairs, work = query_oriented.lookup_pairs(point)
        assert work["endomorphism"] == (0 if point.is_zero() else 2)
        assert work["table_probe"] == 5
        assert all(left.point + right.point == point for left, right in pairs)


test_closed_count_and_exact_lookup_coverage()
test_quotient_solver_reaches_the_same_exact_rank()
test_invalid_window_remains_rejected_through_generic_lookup()
test_query_oriented_index_matches_every_exact_lookup()
print("symmetry_quotient_mcollapse Sage tests: PASS")
