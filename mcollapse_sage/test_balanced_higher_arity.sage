from itertools import combinations_with_replacement

load("mcollapse_sage/rank_guided_mcollapse.sage")
load("mcollapse_sage/symmetry_quotient_mcollapse.sage")
load("mcollapse_sage/balanced_higher_arity.sage")
load("mcollapse_sage/equal_work_mcollapse.sage")


def brute_relation_rows(prepared, target, relation_arity):
    rows = set()
    terms = prepared.terms
    for indices in combinations_with_replacement(
        range(len(terms)), relation_arity
    ):
        selected = tuple(terms[index] for index in indices)
        total = prepared.fixture.E(0)
        for term in selected:
            total += term.point
        if total == target.point:
            rows.add(sparse_terms_row_key(prepared.fixture, selected))
    return rows


def test_phase_lift_matches_every_unordered_pair_row():
    fixture = find_cm_fixture(7, 11)
    reference = PreparedWeightedMCollapse(fixture, 12)
    quotient = PreparedSymmetryQuotientMCollapse(fixture, 12)
    entries, work = lift_exact_pair_entries(quotient)
    assert len(entries) == 10081
    assert len({entry.row_key for entry in entries}) == 10081
    assert len(entries) == unordered_pair_row_count(reference)
    assert all(
        entry.terms[0].point + entry.terms[1].point == entry.point
        for entry in entries
    )
    assert work["endomorphism"] == 2 * sum(
        key != (-1, -1) for key in quotient.quotient_index.index
    )


def test_balanced_generators_match_independent_full_enumeration():
    fixture = find_cm_fixture(7, 19)
    prepared = PreparedSymmetryQuotientMCollapse(fixture, 2)
    _Q, targets = make_public_targets(fixture, 2, 901)
    generator = BalancedHigherArityGenerator(prepared)
    for relation_arity in (4, 5):
        balanced = generator.generate(targets[0], relation_arity)
        balanced_rows = {relation.row_key for relation in balanced.relations}
        reference_rows = brute_relation_rows(
            prepared, targets[0], relation_arity
        )
        assert balanced_rows == reference_rows
        assert balanced.stats["group_failures"] == 0
        assert balanced.stats["unique_full_rows"] == len(reference_rows)


def test_balanced_generators_accept_promoted_query_index():
    fixture = find_cm_fixture(7, 31)
    prepared = PreparedQueryOrientedSymmetryQuotientMCollapse(fixture, 2)
    _Q, targets = make_public_targets(fixture, 2, 907)
    generator = BalancedHigherArityGenerator(prepared)
    for relation_arity in (4, 5):
        balanced = generator.generate(targets[0], relation_arity)
        assert {relation.row_key for relation in balanced.relations} == brute_relation_rows(
            prepared, targets[0], relation_arity
        )


def test_balanced_limit_is_exact_and_deterministic():
    fixture = find_cm_fixture(8, 23)
    prepared = PreparedSymmetryQuotientMCollapse(fixture, 4)
    _Q, targets = make_public_targets(fixture, 2, 777)
    generator = BalancedHigherArityGenerator(prepared)
    first = generator.generate(targets[0], 5, max_unique=25)
    repeated = generator.generate(targets[0], 5, max_unique=25)
    assert len(first.relations) == 25
    assert [relation.row_key for relation in first.relations] == [
        relation.row_key for relation in repeated.relations
    ]
    assert first.work == repeated.work
    emitted_work = [work_total(relation.generation_work) for relation in first.relations]
    assert emitted_work == sorted(emitted_work)
    assert emitted_work[-1] == work_total(first.work)


def test_equal_work_runner_is_exact_and_deterministic():
    fixture = find_cm_fixture(8, 37)
    prepared = PreparedSymmetryQuotientMCollapse(fixture, 4)
    _Q, targets = make_public_targets(fixture, 4, 881)
    for relation_arity in (4, 5):
        first = run_balanced_equal_work(
            prepared, targets, relation_arity, 50000, 991
        )
        repeated = run_balanced_equal_work(
            prepared, targets, relation_arity, 50000, 991
        )
        assert first.state.basis.rank == prepared.width
        assert first.state.verification_failures == 0
        assert first.state.rank_events == repeated.state.rank_events
        assert first.state.variable_work == repeated.state.variable_work
        assert first.generation_stats == repeated.generation_stats


test_phase_lift_matches_every_unordered_pair_row()
test_balanced_generators_match_independent_full_enumeration()
test_balanced_generators_accept_promoted_query_index()
test_balanced_limit_is_exact_and_deterministic()
test_equal_work_runner_is_exact_and_deterministic()
print("balanced_higher_arity Sage tests: PASS")
