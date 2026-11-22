load("mcollapse_sage/rank_guided_mcollapse.sage")


def test_online_basis_against_sage_matrix():
    K = GF(13)
    rows = (
        {0: K(1), 1: K(2)},
        {0: K(2), 1: K(4)},
        {1: K(1), 2: K(3)},
        {0: K(4), 2: K(1), 3: K(2)},
    )
    basis = OnlineSparseBasis(K, 4)
    dense = []
    observed = []
    for sparse in rows:
        dense.append(vector(K, [sparse.get(i, 0) for i in range(4)]))
        residual, _ = basis.reduce(sparse)
        if residual:
            basis.commit(residual)
        observed.append(basis.rank)
        assert basis.rank == matrix(K, dense).rank()
    assert observed == [1, 1, 2, 3]


def test_online_dual_syndrome_matches_exact_rank_increment():
    K = GF(13)
    rows = (
        {0: K(1), 1: K(2)},
        {0: K(2), 1: K(4)},
        {1: K(1), 2: K(3)},
        {0: K(4), 2: K(1), 3: K(2)},
    )
    dual = OnlineDualBasis(K, 4)
    accepted = []
    for sparse in rows:
        before = matrix(K, accepted).rank() if accepted else 0
        dense = vector(K, [sparse.get(i, 0) for i in range(4)])
        after = matrix(K, accepted + [dense]).rank()
        syndrome, _ = dual.syndrome(sparse)
        assert bool(any(syndrome)) == (after > before)
        if any(syndrome):
            dual.commit(sparse, syndrome)
            accepted.append(dense)
            assert dual.rank == after
            assert dual.annihilates(accepted)
    assert dual.rank == 3
    assert dual.nullity == 1


def test_tiny_fixture_and_weighted_catalog():
    fixture = CMFixture("tiny", 7, 3, 13, (1, 2), 2, 3, 0)
    prepared = PreparedWeightedMCollapse(fixture, 2)
    assert prepared.width == 3
    assert len(prepared.terms) == 24
    assert len(prepared.terms)^2 == work_total({
        name: prepared.preparation_work[name] if name == "table_probe" else 0
        for name in WORK_FIELDS
    })
    for term in prepared.terms:
        representative = prepared.representatives[term.anchor]
        assert term.point == ZZ(term.coefficient) * representative


def test_policy_smoke_and_exact_verification():
    fixture = CMFixture("tiny", 7, 3, 13, (1, 2), 2, 3, 0)
    prepared = PreparedWeightedMCollapse(fixture, 2)
    _, targets = make_public_targets(fixture, 8, 19)
    states = {}
    for policy in (
        "weighted-baseline",
        "rank-filtered",
        "rank-saturated",
        "rank-guided",
        "dual-saturated",
        "hybrid-saturated",
        "adaptive-dual",
    ):
        state = run_policy(prepared, targets, policy, 3000, 41)
        states[policy] = state
        assert state.verification_failures == 0
        assert state.basis.rank == matrix(fixture.K, state.basis.dense_rows()).rank()
        assert state.basis.rank <= prepared.width
        assert (
            state.spent() >= 3000
            or state.processed_targets == len(targets)
            or state.basis.rank == prepared.width
        )
    assert states["rank-saturated"].basis.rank == prepared.width
    assert states["rank-guided"].basis.rank == prepared.width
    assert states["dual-saturated"].basis.rank == prepared.width
    assert states["hybrid-saturated"].basis.rank == prepared.width
    assert states["adaptive-dual"].basis.rank == prepared.width
    assert states["dual-saturated"].dual.nullity == 0
    assert states["hybrid-saturated"].dual_initializations == 1
    streaming = run_policy(
        prepared, targets, "rank-saturated", 3000, 41, window_size=1
    )
    assert streaming.rank_events == states["rank-saturated"].rank_events
    assert streaming.variable_work == states["rank-saturated"].variable_work
    guided_window = run_policy(
        prepared, targets, "rank-guided", 3000, 41, window_size=96
    )
    assert guided_window.rank_events == states["rank-guided"].rank_events
    assert guided_window.variable_work == states["rank-guided"].variable_work


def test_sage_fixture_search_hits_exact_bit_band():
    fixture = find_cm_fixture(7, 11)
    assert fixture.n.nbits() == 7
    assert fixture.phi(fixture.G) == ZZ(fixture.eigenvalue) * fixture.G


test_online_basis_against_sage_matrix()
test_online_dual_syndrome_matches_exact_rank_increment()
test_tiny_fixture_and_weighted_catalog()
test_policy_smoke_and_exact_verification()
test_sage_fixture_search_hits_exact_bit_band()
print("rank_guided_mcollapse Sage tests: PASS")
