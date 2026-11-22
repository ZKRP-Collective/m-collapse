import unittest

from mcollapse.curves import generate_fixture, tiny_fixture
from mcollapse.factor_base import BudgetType, Mode, build_comparison_bases
from mcollapse.linear import (
    assert_glv_mcollapse_equivalent,
    canonical_row_space,
    eliminate_glv_coefficients,
    eliminate_glv_columns_with_rhs,
    glv_constraint_rows,
    incremental_ranks,
    rank_mod,
    rref_mod,
)
from mcollapse.relations import prepare_search


class ModularLinearAlgebraTests(unittest.TestCase):
    def test_rref_and_rank_are_exact_and_deterministic_over_f13(self):
        rows = [
            (1, 2, 3, 4),
            (2, 4, 6, 8),
            (0, 1, 1, 0),
            (0, 0, 1, 1),
            (13, -13, 26, 0),
        ]
        expected = [
            (1, 0, 0, 3),
            (0, 1, 0, 12),
            (0, 0, 1, 1),
        ]

        self.assertEqual(rref_mod(rows, 13), expected)
        self.assertEqual(rref_mod(rows, 13), expected)
        self.assertEqual(rank_mod(rows, 13), 3)
        self.assertEqual(canonical_row_space(rows, 13), tuple(expected))

    def test_empty_zero_width_zero_and_rectangular_matrices_are_supported(self):
        self.assertEqual(rref_mod([], 13), [])
        self.assertEqual(rank_mod([], 13), 0)
        self.assertEqual(canonical_row_space([], 13), ())
        self.assertEqual(rref_mod([(), ()], 13), [])
        self.assertEqual(rref_mod([(0, 0, 0), (13, -26, 39)], 13), [])
        self.assertEqual(
            rref_mod([(1, 2, 3), (0, 1, 1)], 13),
            [(1, 0, 1), (0, 1, 1)],
        )

    def test_rows_require_equal_widths_and_integer_cells(self):
        invalid = (
            [(1, 2), (3,)],
            [(1, 2.0)],
            [(True, 0)],
            [(1, "2")],
            [1],
            1,
        )
        for rows in invalid:
            with self.subTest(rows=rows):
                with self.assertRaises((TypeError, ValueError)):
                    rref_mod(rows, 13)

    def test_rref_rank_and_elimination_are_exact_over_f2(self):
        rows = [(1, 1, 0), (1, 0, 1), (0, 1, 1), (2, -2, 4)]

        self.assertEqual(
            rref_mod(rows, 2), [(1, 0, 1), (0, 1, 1)]
        )
        self.assertEqual(rank_mod(rows, 2), 2)
        self.assertEqual(
            eliminate_glv_columns_with_rhs([(1, 1, 0, 3)], 1, 2),
            [(0, 1)],
        )
        self.assertEqual(
            eliminate_glv_coefficients([(1, 0, 1)], 1, 2), [(0,)]
        )

    def test_modulus_must_be_a_prime_integer(self):
        for modulus in (4, 6, 9, 1, 0, -13, 13.0, True, "13"):
            with self.subTest(modulus=modulus):
                with self.assertRaises((TypeError, ValueError)):
                    rank_mod([(1,)], modulus)

    def test_incremental_ranks_agree_with_every_batch_prefix(self):
        rows = [
            (1, 2, 3),
            (2, 4, 6),
            (0, 1, 1),
            (0, 0, 0),
            (0, 0, 1),
        ]
        ranks = incremental_ranks(rows, 13)

        self.assertEqual(ranks, (1, 1, 2, 2, 3))
        self.assertEqual(
            ranks,
            tuple(rank_mod(rows[:end], 13) for end in range(1, len(rows) + 1)),
        )
        self.assertEqual(ranks[-1], rank_mod(rows, 13))
        self.assertEqual(incremental_ranks([], 13), ())


class GlvEliminationTests(unittest.TestCase):
    def test_constraint_rows_use_physical_e0_e1_e2_columns_per_orbit(self):
        self.assertEqual(
            glv_constraint_rows(2, 3, 13),
            [
                (10, 1, 0, 0, 0, 0, 0),
                (4, 0, 1, 0, 0, 0, 0),
                (0, 0, 0, 10, 1, 0, 0),
                (0, 0, 0, 4, 0, 1, 0),
            ],
        )
        self.assertEqual(
            glv_constraint_rows(1, 3, 13, include_rhs=False),
            [(10, 1, 0), (4, 0, 1)],
        )

    def test_invalid_orbit_count_and_constraint_options_are_rejected(self):
        for orbit_count in (0, -1, 1.0, True, "1"):
            with self.subTest(orbit_count=orbit_count):
                with self.assertRaises((TypeError, ValueError)):
                    glv_constraint_rows(orbit_count, 3, 13)
        for include_rhs in (0, 1, None, "yes"):
            with self.subTest(include_rhs=include_rhs):
                with self.assertRaises(TypeError):
                    glv_constraint_rows(1, 3, 13, include_rhs=include_rhs)
        with self.assertRaises(TypeError):
            glv_constraint_rows(1, 3.0, 13)

    def test_synthetic_rows_eliminate_exactly_and_constraints_become_zero(self):
        expanded = [(1, 0, 0, 4), (0, 1, 0, 7), (0, 0, 1, 9)]
        constraints = glv_constraint_rows(1, 3, 13)
        expected = [(1, 4), (3, 7), (9, 9)]

        self.assertEqual(
            eliminate_glv_columns_with_rhs(expanded, 3, 13), expected
        )
        self.assertEqual(
            eliminate_glv_columns_with_rhs(constraints, 3, 13),
            [(0, 0), (0, 0)],
        )
        self.assertEqual(
            canonical_row_space(
                eliminate_glv_columns_with_rhs(expanded + constraints, 3, 13),
                13,
            ),
            canonical_row_space(expected, 13),
        )

    def test_rhs_and_coefficient_only_elimination_are_explicit_and_canonical(self):
        self.assertEqual(
            eliminate_glv_columns_with_rhs(
                [(14, -1, 15, 2, 3, 4, 27)], 3, 13
            ),
            [(3, 8, 1)],
        )
        self.assertEqual(
            eliminate_glv_coefficients([(14, -1, 15, 2, 3, 4)], 3, 13),
            [(3, 8)],
        )
        self.assertEqual(
            eliminate_glv_coefficients([(1, 0, 0), (0, 1, 0), (0, 0, 1)], 3, 13),
            [(1,), (3,), (9,)],
        )
        with self.assertRaises(ValueError):
            eliminate_glv_columns_with_rhs([(1, 2, 3)], 3, 13)
        with self.assertRaises(ValueError):
            eliminate_glv_coefficients([(1, 2, 3, 4)], 3, 13)

    def test_elimination_rejects_malformed_rows_instead_of_guessing(self):
        cases = (
            (eliminate_glv_columns_with_rhs, [(1, 0, 0, 0), (1, 0, 0)]),
            (eliminate_glv_coefficients, [(1, 0, 0), (1, 0)]),
            (eliminate_glv_columns_with_rhs, [(1, 0.0, 0, 0)]),
            (eliminate_glv_coefficients, [(1, True, 0)]),
        )
        for function, rows in cases:
            with self.subTest(function=function.__name__, rows=rows):
                with self.assertRaises((TypeError, ValueError)):
                    function(rows, 3, 13)

    def test_equivalence_assertion_compares_canonical_compact_row_spaces(self):
        expanded = [(1, 0, 0, 4), (2, 0, 0, 8)]
        compact = [(1, 4), (3, 12)]

        self.assertIsNone(
            assert_glv_mcollapse_equivalent(
                expanded,
                compact,
                3,
                13,
                has_rhs=True,
                orbit_count=1,
            )
        )
        self.assertIsNone(
            assert_glv_mcollapse_equivalent(
                [(1, 0, 0), (0, 1, 0)],
                [(1,), (3,)],
                3,
                13,
                has_rhs=False,
                orbit_count=1,
            )
        )

    def test_explicit_shape_allows_zero_row_vs_empty_and_both_empty(self):
        self.assertIsNone(
            assert_glv_mcollapse_equivalent(
                [(0, 0, 0)],
                [],
                3,
                13,
                has_rhs=False,
                orbit_count=1,
            )
        )
        self.assertIsNone(
            assert_glv_mcollapse_equivalent(
                [],
                [],
                3,
                13,
                has_rhs=True,
                orbit_count=2,
            )
        )

    def test_equivalence_rejects_wrong_width_zero_rows_before_reduction(self):
        cases = (
            ([(0, 0)], [], False, 1),
            ([], [(0, 0)], False, 1),
            ([(0, 0, 0)], [(0, 0)], True, 1),
        )
        for expanded, compact, has_rhs, orbit_count in cases:
            with self.subTest(
                expanded=expanded, compact=compact, has_rhs=has_rhs
            ):
                with self.assertRaises(ValueError):
                    assert_glv_mcollapse_equivalent(
                        expanded,
                        compact,
                        3,
                        13,
                        has_rhs=has_rhs,
                        orbit_count=orbit_count,
                    )

    def test_equivalence_requires_a_positive_integer_orbit_count(self):
        for orbit_count in (0, -1, 1.0, True, "1"):
            with self.subTest(orbit_count=orbit_count):
                with self.assertRaises((TypeError, ValueError)):
                    assert_glv_mcollapse_equivalent(
                        [],
                        [],
                        3,
                        13,
                        has_rhs=False,
                        orbit_count=orbit_count,
                    )

    def test_equivalence_mismatch_raises_the_registered_exact_error(self):
        with self.assertRaisesRegex(
            ArithmeticError, "^GLV/MCOLLAPSE row-space mismatch$"
        ):
            assert_glv_mcollapse_equivalent(
                [(1, 0, 0, 4)],
                [(1, 5)],
                3,
                13,
                has_rhs=True,
                orbit_count=1,
            )


class GlvTask4IntegrationTests(unittest.TestCase):
    def test_all_tiny_nonidentity_targets_have_equal_nonempty_relation_spaces(self):
        fixture = tiny_fixture()
        bases = build_comparison_bases(
            fixture, BudgetType.EQUAL_RAW, 6, 31
        )
        glv_base = bases[Mode.GLV_EXPANDED]
        mcollapse_base = bases[Mode.MCOLLAPSE_UNWEIGHTED]
        glv = prepare_search(fixture, glv_base, 2, 10_000)
        mcollapse = prepare_search(fixture, mcollapse_base, 2, 10_000)
        mcollapse_column = {
            representative: index
            for index, representative in enumerate(mcollapse_base.representatives)
        }

        self.assertEqual(
            set(glv_base.representatives), set(mcollapse_base.representatives)
        )
        constraints = glv_constraint_rows(
            len(glv_base.representatives), fixture.eigenvalue, fixture.n
        )
        for scalar in range(1, fixture.n):
            target = fixture.curve.mul(scalar, fixture.generator)
            glv_result = glv.search(target)
            mcollapse_result = mcollapse.search(target)
            self.assertTrue(glv_result.relations, scalar)
            self.assertTrue(mcollapse_result.relations, scalar)
            expanded_rows = [
                relation.normalized_row + (0,)
                for relation in glv_result.relations
            ] + constraints
            compact_rows = [
                tuple(
                    relation.normalized_row[mcollapse_column[representative]]
                    for representative in glv_base.representatives
                )
                + (0,)
                for relation in mcollapse_result.relations
            ]

            assert_glv_mcollapse_equivalent(
                expanded_rows,
                compact_rows,
                fixture.eigenvalue,
                fixture.n,
                has_rhs=True,
                orbit_count=len(glv_base.representatives),
            )

    def test_generated_fixture_smoke_is_fast(self):
        fixture = generate_fixture(7, 20260804)
        expanded = [(2, 0, 0, 3, 0, 0)]
        compact = [(2, 3)]

        assert_glv_mcollapse_equivalent(
            expanded,
            compact,
            fixture.eigenvalue,
            fixture.n,
            has_rhs=False,
            orbit_count=2,
        )
        self.assertEqual(
            eliminate_glv_coefficients(
                glv_constraint_rows(
                    2,
                    fixture.eigenvalue,
                    fixture.n,
                    include_rhs=False,
                ),
                fixture.eigenvalue,
                fixture.n,
            ),
            [(0, 0)] * 4,
        )


if __name__ == "__main__":
    unittest.main()
