load("mcollapse_sage/rank_guided_mcollapse.sage")
load("mcollapse_sage/symmetry_quotient_mcollapse.sage")
load("mcollapse_sage/circuit_mcollapse.sage")


def test_anchor_circuit_solves_exact_public_dlp():
    fixture = find_cm_fixture(8, 67)
    prepared = PreparedQueryOrientedSymmetryQuotientMCollapse(fixture, 4)
    Q, targets = make_public_targets(fixture, 12, 1701)
    for round_robin in (False, True):
        result = run_circuit_mcollapse(
            prepared, targets, 100000, 1901, round_robin=round_robin
        )
        assert result.solved
        assert ZZ(result.recovered) * fixture.G == Q
        assert result.state.verification_failures == 0
        assert result.basis.rank <= prepared.width
        audit = result.exact_rank_audit()
        assert audit["online_anchor_rank_matches"]
        assert audit["system_consistent"]
        assert audit["target_column_adds_rank"]


def test_circuit_basis_detects_direct_target_equation():
    K = GF(101)
    basis = OnlineAnchorCircuitBasis(K, 3)
    first = basis.reduce({0: 1, 1: 2}, -3, 7)
    basis.commit(first[0], first[1], first[2])
    second = basis.reduce({0: 2, 1: 4}, -11, 19)
    assert not second[0]
    recovered = second[2] / second[1]
    assert recovered == K(5) / K(-5)


def test_random_public_targets_are_distinct_and_exact():
    fixture = find_cm_fixture(8, 71)
    Q, targets = make_distinct_random_public_targets(fixture, 32, 2101)
    assert len({point_key(target.point) for target in targets}) == len(targets)
    assert all(
        target.point == target.u * fixture.G + target.v * Q
        for target in targets
    )


test_anchor_circuit_solves_exact_public_dlp()
test_circuit_basis_detects_direct_target_equation()
test_random_public_targets_are_distinct_and_exact()
print("circuit_mcollapse Sage tests: PASS")
