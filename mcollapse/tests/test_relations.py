import ast
import math
import unittest
from dataclasses import FrozenInstanceError, fields, replace
from inspect import signature
from pathlib import Path

from mcollapse.costs import (
    AFFINE_ADD,
    AFFINE_DOUBLE,
    CANCELLATION,
    IDENTITY_NOOP,
    NORMALIZE,
    PHI,
    PROBE,
    CostVector,
)
from mcollapse.curves import generate_fixture, tiny_fixture
from mcollapse.factor_base import BudgetType, FactorBase, Mode, build_comparison_bases
from mcollapse.orbits import canonicalize, phi_power
from mcollapse.relations import (
    CombinationBranch,
    PreparedSearch,
    Relation,
    SearchResult,
    Term,
    TermCatalog,
    aggregate_candidates,
    aggregate_cost,
    build_terms,
    combine_points,
    find_relations,
    normalize_row,
    prepare_search,
    relation_from_terms,
    verify_relation,
)


class CostVectorTests(unittest.TestCase):
    def test_registered_vectors_record_complete_affine_formula_assumptions(self):
        self.assertEqual(
            AFFINE_ADD,
            CostVector(fp_mul=2, fp_square=1, fp_inv=1, point_add=1),
        )
        self.assertEqual(
            AFFINE_DOUBLE,
            CostVector(fp_mul=3, fp_square=2, fp_inv=1, point_double=1),
        )
        self.assertEqual(PHI, CostVector(fp_mul=1, endomorphism=1))
        self.assertEqual(PROBE, CostVector(table_probe=1))
        self.assertEqual(NORMALIZE, CostVector(normalization=1))
        self.assertEqual(CANCELLATION, CostVector(point_add=1))
        self.assertEqual(IDENTITY_NOOP, CostVector())

    def test_arithmetic_is_componentwise_and_immutable(self):
        left = CostVector(fp_mul=2, point_add=1)
        right = CostVector(fp_square=3, table_probe=4)

        self.assertEqual(
            left + right,
            CostVector(fp_mul=2, fp_square=3, point_add=1, table_probe=4),
        )
        self.assertEqual(3 * left, CostVector(fp_mul=6, point_add=3))
        self.assertEqual(right * 2, CostVector(fp_square=6, table_probe=8))
        with self.assertRaises(FrozenInstanceError):
            left.fp_mul = 4

    def test_counts_and_arithmetic_operands_are_validated(self):
        for value in (-1, 1.0, True, "1"):
            with self.subTest(value=value):
                with self.assertRaises((TypeError, ValueError)):
                    CostVector(fp_mul=value)
        vector = CostVector(fp_mul=1)
        for factor in (-1, 1.0, True, "2"):
            with self.subTest(factor=factor):
                with self.assertRaises((TypeError, ValueError)):
                    vector * factor
        with self.assertRaises(TypeError):
            vector + 1

    def test_calibration_requires_exact_finite_nonnegative_numeric_weights(self):
        names = tuple(field.name for field in fields(CostVector))
        weights = {name: index + 0.5 for index, name in enumerate(names)}
        vector = CostVector(fp_mul=2, fp_inv=1, table_probe=3, normalization=4)
        expected = (
            2 * weights["fp_mul"]
            + weights["fp_inv"]
            + 3 * weights["table_probe"]
            + 4 * weights["normalization"]
        )

        self.assertEqual(vector.calibrated(weights), expected)
        invalid = (
            {name: 1 for name in names[:-1]},
            {**{name: 1 for name in names}, "extra": 1},
            {**{name: 1 for name in names}, names[0]: -1},
            {**{name: 1 for name in names}, names[0]: math.inf},
            {**{name: 1 for name in names}, names[0]: True},
        )
        for candidate in invalid:
            with self.subTest(candidate=candidate):
                with self.assertRaises((TypeError, ValueError)):
                    vector.calibrated(candidate)


class TermConstructionTests(unittest.TestCase):
    def setUp(self):
        self.fixture = tiny_fixture()
        self.bases = build_comparison_bases(
            self.fixture, BudgetType.EQUAL_UNKNOWNS, 2, 17
        )

    def test_mode_specific_points_columns_and_coefficients_are_exact(self):
        fixture = self.fixture
        for mode, base in self.bases.items():
            with self.subTest(mode=mode):
                actual = build_terms(fixture, base)
                self.assertIsInstance(actual, TermCatalog)
                expected = []
                for index, representative in enumerate(base.representatives):
                    for source in base.allowed_points(fixture, representative):
                        if mode is Mode.CLASSIC:
                            expected.append((source, index, 1))
                            continue
                        if mode is Mode.NEGATION:
                            sign = 1 if source == representative else -1
                            expected.append((source, index, sign))
                            continue
                        encoding = canonicalize(fixture, source)
                        if mode is Mode.GLV_EXPANDED:
                            expected.append(
                                (
                                    source,
                                    3 * index + encoding.exponent,
                                    encoding.sign,
                                )
                            )
                            continue
                        multiplier = (
                            encoding.sign
                            * pow(fixture.eigenvalue, encoding.exponent, fixture.n)
                        ) % fixture.n
                        weights = (-2, -1, 1, 2) if mode is Mode.MCOLLAPSE_WEIGHTED else (1,)
                        for weight in weights:
                            point = fixture.curve.mul(weight, source)
                            if point is not None:
                                expected.append(
                                    (point, index, weight * multiplier % fixture.n)
                                )
                expected = tuple(dict.fromkeys(expected))
                observed = tuple(
                    (term.point, term.column, term.coefficient)
                    for term in actual.terms
                )
                self.assertEqual(observed, expected)
                for term in actual.terms:
                    representative = base.representatives[term.column // 3]
                    if mode is Mode.GLV_EXPANDED:
                        physical_point = phi_power(
                            fixture, representative, term.column % 3
                        )
                    else:
                        representative = base.representatives[term.column]
                        physical_point = representative
                    coefficient_point = fixture.curve.mul(
                        term.coefficient, physical_point
                    )
                    self.assertEqual(coefficient_point, term.point)

    def test_term_costs_include_phi_and_weight_two_doubling(self):
        fixture = self.fixture
        for mode in (Mode.GLV_EXPANDED, Mode.MCOLLAPSE_UNWEIGHTED, Mode.MCOLLAPSE_WEIGHTED):
            base = self.bases[mode]
            for term in build_terms(fixture, base).terms:
                index = term.column // 3 if mode is Mode.GLV_EXPANDED else term.column
                representative = base.representatives[index]
                possible = []
                for source in base.allowed_points(fixture, representative):
                    encoding = canonicalize(fixture, source)
                    weights = (-2, -1, 1, 2) if mode is Mode.MCOLLAPSE_WEIGHTED else (1,)
                    for weight in weights:
                        if fixture.curve.mul(weight, source) == term.point:
                            possible.append(
                                PHI * encoding.exponent
                                + (AFFINE_DOUBLE if abs(weight) == 2 else CostVector())
                            )
                self.assertIn(term.cost, possible)

    def test_equal_raw_terms_never_expand_past_per_representative_allowlist(self):
        fixture = self.fixture
        bases = build_comparison_bases(fixture, BudgetType.EQUAL_RAW, 2, 17)
        for mode, base in bases.items():
            with self.subTest(mode=mode):
                terms = build_terms(fixture, base)
                for term in terms.terms:
                    index = term.column // 3 if mode is Mode.GLV_EXPANDED else term.column
                    allowed = base.allowed_points(fixture, base.representatives[index])
                    weights = (-2, -1, 1, 2) if mode is Mode.MCOLLAPSE_WEIGHTED else (1,)
                    self.assertTrue(
                        any(
                            fixture.curve.mul(weight, source) == term.point
                            for source in allowed
                            for weight in weights
                        )
                    )

    def test_weighted_term_aliases_are_deduplicated_deterministically(self):
        base = self.bases[Mode.MCOLLAPSE_WEIGHTED]
        first = build_terms(self.fixture, base)
        second = build_terms(self.fixture, base)
        triples = tuple(
            (term.point, term.column, term.coefficient) for term in first.terms
        )

        self.assertEqual(first, second)
        self.assertEqual(len(triples), len(set(triples)))
        self.assertNotIn(None, (term.point for term in first.terms))

    def test_weighted_budget_32_charges_all_attempts_before_deduplication(self):
        fixture = generate_fixture(10, 20260804)
        base = build_comparison_bases(
            fixture, BudgetType.EQUAL_UNKNOWNS, 32, 20260804
        )[Mode.MCOLLAPSE_WEIGHTED]
        catalog = build_terms(fixture, base)

        self.assertEqual(catalog.attempted_terms, 768)
        self.assertEqual(len(catalog.terms), 384)
        self.assertEqual(
            catalog.construction_cost,
            PHI * 768 + AFFINE_DOUBLE * 384,
        )
        with self.assertRaises(FrozenInstanceError):
            catalog.attempted_terms = 0

    def test_physical_width_differs_only_for_expanded_glv(self):
        fixture = self.fixture
        glv = self.bases[Mode.GLV_EXPANDED]
        mcollapse = self.bases[Mode.MCOLLAPSE_UNWEIGHTED]
        glv_relation = relation_from_terms(
            fixture,
            build_terms(fixture, glv).terms[:1],
            3 * len(glv.representatives),
        )
        mcollapse_relation = relation_from_terms(
            fixture,
            build_terms(fixture, mcollapse).terms[:1],
            len(mcollapse.representatives),
        )

        self.assertEqual(len(glv_relation.normalized_row), 3 * len(glv.representatives))
        self.assertEqual(len(mcollapse_relation.normalized_row), len(mcollapse.representatives))


class RelationSearchTests(unittest.TestCase):
    def test_point_combination_classifies_and_charges_every_branch(self):
        fixture = tiny_fixture()
        point = fixture.generator
        other = fixture.curve.mul(2, point)
        cases = (
            (
                point,
                other,
                CombinationBranch.ORDINARY_ADD,
                AFFINE_ADD,
                fixture.curve.add(point, other),
            ),
            (
                point,
                point,
                CombinationBranch.DOUBLING,
                AFFINE_DOUBLE,
                fixture.curve.add(point, point),
            ),
            (
                point,
                fixture.curve.neg(point),
                CombinationBranch.INVERSE_CANCELLATION,
                CANCELLATION,
                None,
            ),
            (
                None,
                point,
                CombinationBranch.IDENTITY_NOOP,
                IDENTITY_NOOP,
                point,
            ),
        )
        for left, right, branch, cost, expected in cases:
            with self.subTest(branch=branch):
                combination = combine_points(fixture, left, right)
                self.assertEqual(combination.branch, branch)
                self.assertEqual(combination.cost, cost)
                self.assertEqual(combination.point, expected)

    def test_normalization_only_aggregates_coefficients_modulo_n(self):
        fixture = tiny_fixture()
        terms = (
            Term(fixture.generator, 1, -1, CostVector()),
            Term(fixture.generator, 0, 2, CostVector()),
            Term(fixture.generator, 1, fixture.n + 3, CostVector()),
        )
        relation = relation_from_terms(fixture, terms, 3)

        self.assertEqual(normalize_row((2, fixture.n + 2, 0), fixture.n), (2, 2, 0))
        self.assertEqual(relation.normalized_row, (2, 2, 0))
        self.assertEqual(
            relation.cost,
            AFFINE_DOUBLE + AFFINE_ADD + NORMALIZE * 3,
        )

    def test_verification_requires_catalog_terms_canonical_row_and_row_point_target(self):
        fixture = tiny_fixture()
        point = fixture.generator
        base = FactorBase(Mode.CLASSIC, (point,), (point,), 1, 0)
        prepared = prepare_search(fixture, base, 2, 4)
        allowed = prepared.catalog.terms[0]
        relation = relation_from_terms(fixture, (allowed,), 1)
        corrupted = replace(relation, normalized_row=(2,))
        noncanonical = replace(
            relation, normalized_row=(relation.normalized_row[0] + fixture.n,)
        )
        forged = relation_from_terms(
            fixture, (Term(point, 0, 2, CostVector()),), 1
        )
        altered_catalog_term = relation_from_terms(
            fixture, (replace(allowed, cost=PHI),), 1
        )

        self.assertTrue(verify_relation(prepared, relation, point))
        self.assertFalse(verify_relation(prepared, corrupted, point))
        self.assertFalse(verify_relation(prepared, noncanonical, point))
        self.assertFalse(verify_relation(prepared, forged, point))
        self.assertFalse(verify_relation(prepared, altered_catalog_term, point))

    def test_every_emitted_arity_two_and_three_relation_reconstructs_target(self):
        fixture = tiny_fixture()
        bases = build_comparison_bases(
            fixture, BudgetType.EQUAL_UNKNOWNS, 2, 7
        )
        target = fixture.curve.mul(4, fixture.generator)
        for arity in (2, 3):
            for mode, base in bases.items():
                with self.subTest(arity=arity, mode=mode):
                    prepared, result = find_relations(
                        fixture, base, target, arity=arity, candidate_limit=10_000
                    )
                    self.assertIsInstance(result, SearchResult)
                    self.assertEqual(len(result), len(result.relations))
                    self.assertTrue(
                        all(
                            verify_relation(prepared, relation, target)
                            for relation in result
                        )
                    )

    def test_every_mode_term_matches_its_physical_matrix_basis_column(self):
        fixture = tiny_fixture()
        bases = build_comparison_bases(
            fixture, BudgetType.EQUAL_UNKNOWNS, 2, 19
        )
        for mode, base in bases.items():
            prepared = prepare_search(fixture, base, 2, 10_000)
            expected_basis = (
                tuple(
                    phi_power(fixture, representative, exponent)
                    for representative in base.representatives
                    for exponent in range(3)
                )
                if mode is Mode.GLV_EXPANDED
                else base.representatives
            )
            self.assertEqual(prepared.basis_points, expected_basis)
            for term in prepared.catalog.terms:
                with self.subTest(mode=mode, term=term):
                    relation = relation_from_terms(
                        fixture, (term,), prepared.column_count
                    )
                    self.assertTrue(
                        verify_relation(prepared, relation, term.point)
                    )

    def test_weighted_aliases_dedupe_exact_rows_and_group_results(self):
        fixture = tiny_fixture()
        base = build_comparison_bases(
            fixture, BudgetType.EQUAL_UNKNOWNS, 2, 11
        )[Mode.MCOLLAPSE_WEIGHTED]
        target = fixture.curve.mul(5, fixture.generator)
        prepared, result = find_relations(fixture, base, target, 3, 10_000)
        rows = tuple(relation.normalized_row for relation in result)

        self.assertEqual(len(rows), len(set(rows)))
        self.assertTrue(all(verify_relation(prepared, relation, target) for relation in result))
        repeated_prepared, repeated = find_relations(fixture, base, target, 3, 10_000)
        self.assertEqual(result, repeated)
        self.assertEqual(prepared.preparation_cost, repeated_prepared.preparation_cost)

    def test_candidate_limit_counts_failed_probes_and_repeat_is_deterministic(self):
        fixture = tiny_fixture()
        point = fixture.generator
        base = FactorBase(Mode.CLASSIC, (point,), (point,), 1, 0)
        target = fixture.curve.mul(4, point)
        prepared, first = find_relations(fixture, base, target, 2, 1)
        repeated_prepared, second = find_relations(fixture, base, target, 2, 1)

        self.assertEqual(first, second)
        self.assertEqual(prepared.preparation_candidates, 0)
        self.assertEqual(prepared.preparation_cost, repeated_prepared.preparation_cost)
        self.assertEqual(first.search_candidates, 1)
        self.assertEqual(first.search_cost, AFFINE_ADD + PROBE)
        self.assertFalse(hasattr(first, "preparation_cost"))
        self.assertFalse(hasattr(first, "total_cost"))
        self.assertFalse(hasattr(first, "generation_cost"))
        self.assertFalse(hasattr(first, "candidates"))
        self.assertEqual(first.relations, ())

    def test_duplicate_hits_are_all_counted_before_exact_row_deduplication(self):
        fixture = tiny_fixture()
        left = fixture.generator
        right = fixture.curve.mul(2, left)
        base = FactorBase(Mode.CLASSIC, (left, right), (left, right), 2, 0)
        target = fixture.curve.add(left, right)
        prepared, result = find_relations(fixture, base, target, 2, 8)

        self.assertEqual(len(result.relations), 1)
        self.assertEqual(prepared.preparation_candidates, 2)
        self.assertEqual(result.search_candidates, 4)
        self.assertEqual(prepared.preparation_candidates + result.search_candidates, 6)
        self.assertEqual(
            result.search_cost,
            AFFINE_ADD * 8 + PROBE * 4 + NORMALIZE * 8,
        )

    def test_doubling_hit_ledger_counts_construction_and_both_verification_sums(self):
        fixture = tiny_fixture()
        point = fixture.generator
        base = FactorBase(Mode.CLASSIC, (point,), (point,), 1, 0)
        target = fixture.curve.mul(2, point)
        prepared, result = find_relations(fixture, base, target, 2, 4)

        self.assertEqual(len(result.relations), 1)
        self.assertEqual(
            result.search_cost,
            AFFINE_ADD + AFFINE_DOUBLE * 3 + PROBE * 2 + NORMALIZE * 2,
        )
        self.assertTrue(verify_relation(prepared, result.relations[0], target))

    def test_cancellation_hit_ledger_counts_construction_and_both_verification_sums(self):
        fixture = tiny_fixture()
        point = fixture.generator
        negative = fixture.curve.neg(point)
        target = fixture.curve.mul(2, point)
        base = FactorBase(
            Mode.CLASSIC,
            (point, negative, target),
            (point, negative, target),
            3,
            0,
        )
        prepared, result = find_relations(fixture, base, target, 3, 12)

        self.assertEqual(len(result.relations), 1)
        self.assertEqual(result.search_candidates, 6)
        self.assertEqual(
            result.search_cost,
            AFFINE_ADD * 2
            + CANCELLATION * 10
            + PROBE * 6
            + NORMALIZE * 18,
        )
        self.assertTrue(verify_relation(prepared, result.relations[0], target))

    def test_arity_three_limit_one_builds_no_pairs_and_counts_one_failed_probe(self):
        fixture = tiny_fixture()
        base = build_comparison_bases(
            fixture, BudgetType.EQUAL_UNKNOWNS, 2, 3
        )[Mode.CLASSIC]
        prepared = prepare_search(fixture, base, 3, 1)
        target = fixture.curve.mul(6, fixture.generator)
        result = prepared.search(target)

        self.assertEqual(prepared.preparation_candidates, 0)
        self.assertEqual(result.search_candidates, 1)
        self.assertEqual(prepared.preparation_candidates + result.search_candidates, 1)
        self.assertEqual(result.relations, ())
        self.assertEqual(result.search_cost.table_probe, 1)
        self.assertEqual(
            result.search_cost.point_add
            + result.search_cost.point_double,
            1,
        )

    def test_budget_32_pair_preparation_is_bounded_immutable_and_reusable(self):
        fixture = generate_fixture(10, 20260804)
        base = build_comparison_bases(
            fixture, BudgetType.EQUAL_UNKNOWNS, 32, 20260804
        )[Mode.MCOLLAPSE_WEIGHTED]
        prepared = prepare_search(fixture, base, 3, 32)
        first_target = fixture.generator
        second_target = fixture.curve.mul(2, fixture.generator)
        first = prepared.search(first_target)
        repeated = prepared.search(first_target)
        second = prepared.search(second_target)

        self.assertIsInstance(prepared, PreparedSearch)
        self.assertEqual(prepared.catalog.attempted_terms, 768)
        self.assertEqual(prepared.preparation_candidates, 16)
        self.assertEqual(prepared.preparation_cost.table_probe, 16)
        self.assertEqual(first, repeated)
        self.assertLessEqual(first.search_candidates, 16)
        self.assertLessEqual(second.search_candidates, 16)
        self.assertLessEqual(
            prepared.preparation_candidates + first.search_candidates, 32
        )
        self.assertLessEqual(
            prepared.preparation_candidates + second.search_candidates, 32
        )
        with self.assertRaises(FrozenInstanceError):
            prepared.preparation_candidates = 0

    def test_aggregate_helpers_charge_preparation_once_for_100_targets(self):
        fixture = tiny_fixture()
        point = fixture.generator
        base = FactorBase(Mode.CLASSIC, (point,), (point,), 1, 0)
        prepared = prepare_search(fixture, base, 2, 4)
        targets = tuple(
            fixture.curve.mul(index % (fixture.n - 1) + 1, point)
            for index in range(100)
        )
        results = tuple(prepared.search(target) for target in targets)
        expected_cost = prepared.preparation_cost
        for result in results:
            expected_cost += result.search_cost

        self.assertEqual(aggregate_cost(prepared, results), expected_cost)
        self.assertEqual(
            aggregate_candidates(prepared, results),
            prepared.preparation_candidates
            + sum(result.search_candidates for result in results),
        )
        self.assertNotEqual(
            aggregate_cost(prepared, results),
            prepared.preparation_cost * 100
            + sum((result.search_cost for result in results), CostVector()),
        )

    def test_models_are_frozen_and_invalid_inputs_are_rejected(self):
        fixture = tiny_fixture()
        term = Term(fixture.generator, 0, 1, CostVector())
        relation = relation_from_terms(fixture, (term,), 1)
        base = FactorBase(Mode.CLASSIC, (fixture.generator,), (fixture.generator,), 1, 0)
        prepared = prepare_search(fixture, base, 2, 2)
        result = prepared.search(fixture.generator)
        for instance, attribute, value in (
            (term, "column", 2),
            (relation, "terms", ()),
            (result, "search_candidates", 2),
        ):
            with self.subTest(instance=type(instance).__name__):
                with self.assertRaises(FrozenInstanceError):
                    setattr(instance, attribute, value)

        invalid_terms = (
            (None, 0, 1, CostVector()),
            ((0, 0), -1, 1, CostVector()),
            (fixture.generator, 0, 1.0, CostVector()),
            (fixture.generator, 0, 1, None),
        )
        for arguments in invalid_terms:
            with self.subTest(arguments=arguments):
                with self.assertRaises((TypeError, ValueError)):
                    Term(*arguments)
        with self.assertRaises(ValueError):
            relation_from_terms(fixture, (Term((0, 0), 0, 1, CostVector()),), 1)
        with self.assertRaises(ValueError):
            relation_from_terms(fixture, (Term(fixture.generator, 2, 1, CostVector()),), 1)

        base = FactorBase(Mode.CLASSIC, (fixture.generator,), (fixture.generator,), 1, 0)
        for arity in (1, 4, True):
            with self.subTest(arity=arity):
                with self.assertRaises((TypeError, ValueError)):
                    find_relations(fixture, base, fixture.generator, arity, 1)
        for limit in (0, -1, 1.0, True):
            with self.subTest(limit=limit):
                with self.assertRaises((TypeError, ValueError)):
                    find_relations(fixture, base, fixture.generator, 2, limit)
        for target in (None, (0, 0)):
            with self.subTest(target=target):
                with self.assertRaises(ValueError):
                    find_relations(fixture, base, target, 2, 1)

    def test_public_relation_api_has_no_hidden_data_channel(self):
        functions = (
            normalize_row,
            build_terms,
            combine_points,
            prepare_search,
            aggregate_cost,
            aggregate_candidates,
            relation_from_terms,
            verify_relation,
            find_relations,
        )
        forbidden = {
            "oracle",
            "private",
            "secret",
            "log",
            "scalar",
            "scalar_map",
        }
        for function in functions:
            with self.subTest(function=function.__name__):
                self.assertTrue(
                    forbidden.isdisjoint(signature(function).parameters)
                )

        root = Path(__file__).parents[1]
        for name in ("costs.py", "relations.py"):
            tree = ast.parse((root / name).read_text(encoding="ascii"))
            imports = {
                alias.asname or alias.name.rsplit(".", 1)[-1]
                for node in ast.walk(tree)
                if isinstance(node, (ast.Import, ast.ImportFrom))
                for alias in node.names
            }
            self.assertTrue(forbidden.isdisjoint(imports))

    def test_generated_fixture_smoke(self):
        fixture = generate_fixture(7, 20260804)
        base = build_comparison_bases(
            fixture, BudgetType.EQUAL_UNKNOWNS, 1, 20260804
        )[Mode.MCOLLAPSE_UNWEIGHTED]
        target = fixture.curve.mul(2, fixture.generator)
        prepared, result = find_relations(fixture, base, target, 2, 4)

        self.assertTrue(all(verify_relation(prepared, relation, target) for relation in result))


if __name__ == "__main__":
    unittest.main()
