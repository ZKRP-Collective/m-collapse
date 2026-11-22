import hashlib
import unittest
from dataclasses import FrozenInstanceError, fields, replace
from inspect import signature

from mcollapse.curves import generate_fixture, tiny_fixture
from mcollapse.factor_base import (
    BudgetType,
    FactorBase,
    Mode,
    build_comparison_bases,
    choose_representatives,
    negation_representative,
    point_rank,
    ranked_subgroup_points,
    raw_closure,
    representative_for,
)
from mcollapse.orbits import canonicalize


class EnumAndDataModelTests(unittest.TestCase):
    def test_enum_values_are_exact(self):
        self.assertEqual(
            [mode.value for mode in Mode],
            [
                "classic",
                "negation",
                "glv-expanded",
                "mcollapse-unweighted",
                "mcollapse-weighted",
            ],
        )
        self.assertEqual(
            [budget_type.value for budget_type in BudgetType],
            ["equal-raw", "equal-unknowns", "equal-work"],
        )

    def test_factor_base_is_frozen_and_contains_no_oracle_metadata(self):
        base = FactorBase(Mode.CLASSIC, ((1, 2),), ((1, 2),), 1, 1)
        unmatched_base = FactorBase(Mode.CLASSIC, ((1, 2),), ((1, 2),), 1, 0)

        self.assertIsInstance(base.representatives, tuple)
        self.assertIsInstance(base.raw_points, tuple)
        self.assertEqual(unmatched_base.candidate_limit, 0)
        with self.assertRaises(FrozenInstanceError):
            base.effective_columns = 2
        self.assertTrue(
            {field.name for field in fields(FactorBase)}.isdisjoint(
                {"scalar", "log", "private", "secret", "oracle"}
            )
        )

    def test_factor_base_rejects_empty_or_inconsistent_values(self):
        invalid_arguments = (
            (Mode.CLASSIC, (), ((1, 2),), 1, 1),
            (Mode.CLASSIC, ((1, 2),), (), 1, 1),
            (Mode.CLASSIC, ((1, 2),), ((1, 2),), 2, 1),
            (Mode.CLASSIC, ((1, 2),), ((1, 2),), 1, -1),
            ("classic", ((1, 2),), ((1, 2),), 1, 1),
        )

        for arguments in invalid_arguments:
            with self.subTest(arguments=arguments):
                with self.assertRaises((TypeError, ValueError)):
                    FactorBase(*arguments)

    def test_selection_api_has_no_scalar_or_oracle_data_channel(self):
        expected_parameters = {
            point_rank: ("fixture", "seed", "point"),
            ranked_subgroup_points: ("fixture", "seed"),
            negation_representative: ("fixture", "point"),
            representative_for: ("fixture", "mode", "point"),
            choose_representatives: ("fixture", "mode", "ranked_points", "count"),
            raw_closure: ("fixture", "mode", "representatives"),
            build_comparison_bases: ("fixture", "budget_type", "budget", "seed"),
        }
        forbidden = {"scalar", "log", "private", "secret", "oracle", "scalar_map"}

        for function, parameters in expected_parameters.items():
            with self.subTest(function=function.__name__):
                actual = tuple(signature(function).parameters)
                self.assertEqual(actual, parameters)
                self.assertTrue(forbidden.isdisjoint(actual))

        points = ranked_subgroup_points(tiny_fixture(), 17)
        self.assertTrue(
            all(
                isinstance(point, tuple)
                and len(point) == 2
                and all(isinstance(coordinate, int) for coordinate in point)
                for point in points
            )
        )


class RankingTests(unittest.TestCase):
    def test_point_rank_hashes_registered_compressed_encoding(self):
        fixture = tiny_fixture()
        seed_bytes = b"17"
        domain_and_seed = (
            b"owic-factor-base-rank-v2\x00"
            + len(seed_bytes).to_bytes(4, "big")
            + seed_bytes
        )
        even_expected = hashlib.sha256(domain_and_seed + b"\x02\x01").digest()
        odd_expected = hashlib.sha256(domain_and_seed + b"\x03\x01").digest()

        self.assertEqual(point_rank(fixture, 17, (1, 2)), even_expected)
        self.assertEqual(point_rank(fixture, 17, (1, 5)), odd_expected)
        self.assertNotEqual(
            point_rank(fixture, 17, (1, 2)),
            point_rank(fixture, 18, (1, 2)),
        )
        with self.assertRaises(ValueError):
            point_rank(fixture, 17, None)

    def test_point_rank_uses_field_width_for_generated_fixture(self):
        fixture = generate_fixture(7, 20260804)
        point = fixture.generator
        width = (fixture.p.bit_length() + 7) // 8
        seed_bytes = b"-9"
        compressed = bytes((2 | (point[1] & 1),)) + point[0].to_bytes(width, "big")
        expected = hashlib.sha256(
            b"owic-factor-base-rank-v2\x00"
            + len(seed_bytes).to_bytes(4, "big")
            + seed_bytes
            + compressed
        ).digest()

        self.assertEqual(width, 2)
        self.assertEqual(len(compressed), width + 1)
        self.assertEqual(point_rank(fixture, -9, point), expected)

    def test_ranked_points_are_complete_deterministic_and_seed_dependent(self):
        fixture = tiny_fixture()
        first = ranked_subgroup_points(fixture, 17)
        repeated = ranked_subgroup_points(fixture, 17)
        other_seed = ranked_subgroup_points(fixture, 18)

        self.assertIsInstance(first, tuple)
        self.assertEqual(first, repeated)
        self.assertNotEqual(first, other_seed)
        self.assertEqual(len(first), fixture.n - 1)
        self.assertEqual(len(set(first)), fixture.n - 1)
        self.assertNotIn(None, first)
        self.assertEqual(
            first,
            tuple(
                sorted(
                    first,
                    key=lambda point: (point_rank(fixture, 17, point), point),
                )
            ),
        )
        self.assertTrue(
            all(fixture.curve.mul(fixture.n, point) is None for point in first)
        )


class RepresentativeTests(unittest.TestCase):
    def test_mode_specific_representatives_are_deterministic_quotients(self):
        fixture = tiny_fixture()
        point = fixture.generator

        self.assertEqual(representative_for(fixture, Mode.CLASSIC, point), point)
        self.assertEqual(
            representative_for(fixture, Mode.NEGATION, point),
            min(point, fixture.curve.neg(point)),
        )
        orbit_representative = canonicalize(fixture, point).representative
        for mode in (Mode.GLV_EXPANDED, Mode.MCOLLAPSE_UNWEIGHTED, Mode.MCOLLAPSE_WEIGHTED):
            with self.subTest(mode=mode):
                self.assertEqual(
                    representative_for(fixture, mode, point), orbit_representative
                )

    def test_chosen_representatives_are_unique_and_preserve_first_rank_order(self):
        fixture = tiny_fixture()
        ranked = ranked_subgroup_points(fixture, 23)

        for mode in Mode:
            with self.subTest(mode=mode):
                representatives = choose_representatives(fixture, mode, ranked, 2)
                expected = []
                for point in ranked:
                    representative = representative_for(fixture, mode, point)
                    if representative not in expected:
                        expected.append(representative)
                    if len(expected) == 2:
                        break
                self.assertEqual(representatives, tuple(expected))
                self.assertEqual(len(set(representatives)), 2)

    def test_raw_closure_reconstructs_valid_points_without_identity(self):
        fixture = tiny_fixture()
        ranked = ranked_subgroup_points(fixture, 29)

        for mode in Mode:
            with self.subTest(mode=mode):
                representatives = choose_representatives(fixture, mode, ranked, 2)
                raw_points = raw_closure(fixture, mode, representatives)
                self.assertNotIn(None, raw_points)
                self.assertTrue(
                    all(fixture.curve.is_on_curve(point) for point in raw_points)
                )
                self.assertEqual(
                    {
                        representative_for(fixture, mode, point)
                        for point in raw_points
                    },
                    set(representatives),
                )

    def test_representative_helpers_reject_identity_and_off_curve_points(self):
        fixture = tiny_fixture()

        for point in (None, (0, 0)):
            with self.subTest(point=point):
                with self.assertRaises(ValueError):
                    negation_representative(fixture, point)
                with self.assertRaises(ValueError):
                    representative_for(fixture, Mode.CLASSIC, point)
                with self.assertRaises(ValueError):
                    choose_representatives(fixture, Mode.CLASSIC, (point,), 1)
                with self.assertRaises(ValueError):
                    raw_closure(fixture, Mode.CLASSIC, (point,))


class BudgetTests(unittest.TestCase):
    def test_all_budget_types_return_all_five_modes(self):
        fixture = tiny_fixture()

        for budget_type in BudgetType:
            with self.subTest(budget_type=budget_type):
                bases = build_comparison_bases(fixture, budget_type, 2, 17)
                self.assertEqual(tuple(bases), tuple(Mode))
                self.assertEqual({base.mode for base in bases.values()}, set(Mode))

    def test_equal_unknown_budget_has_exact_identical_effective_columns(self):
        bases = build_comparison_bases(
            tiny_fixture(), BudgetType.EQUAL_UNKNOWNS, 2, 17
        )

        self.assertEqual({base.effective_columns for base in bases.values()}, {2})
        self.assertTrue(all(len(base.representatives) == 2 for base in bases.values()))
        for mode, base in bases.items():
            with self.subTest(mode=mode):
                self.assertEqual(
                    base.raw_points,
                    raw_closure(tiny_fixture(), mode, base.representatives),
                )

    def test_equal_raw_uses_one_ranked_tuple_without_closure_expansion(self):
        fixture = tiny_fixture()
        expected_raw = ranked_subgroup_points(fixture, 17)[:5]
        bases = build_comparison_bases(fixture, BudgetType.EQUAL_RAW, 5, 17)

        self.assertEqual({base.raw_points for base in bases.values()}, {expected_raw})
        self.assertEqual(
            {frozenset(base.raw_points) for base in bases.values()},
            {frozenset(expected_raw)},
        )
        for mode, base in bases.items():
            with self.subTest(mode=mode):
                self.assertEqual(
                    base.representatives,
                    tuple(
                        dict.fromkeys(
                            representative_for(fixture, mode, point)
                            for point in expected_raw
                        )
                    ),
                )
                self.assertEqual(base.effective_columns, len(base.representatives))
                self.assertTrue(set(base.raw_points) <= set(expected_raw))

    def test_equal_raw_allowlists_exactly_the_shared_coverage(self):
        fixture = tiny_fixture()
        bases = build_comparison_bases(fixture, BudgetType.EQUAL_RAW, 5, 17)

        for mode, base in bases.items():
            with self.subTest(mode=mode):
                allowed_by_representative = tuple(
                    base.allowed_points(fixture, representative)
                    for representative in base.representatives
                )
                self.assertTrue(all(allowed_by_representative))
                for representative, allowed_points in zip(
                    base.representatives, allowed_by_representative
                ):
                    self.assertEqual(
                        allowed_points,
                        tuple(
                            point
                            for point in base.raw_points
                            if representative_for(fixture, mode, point)
                            == representative
                        ),
                    )
                self.assertEqual(
                    {
                        point
                        for allowed_points in allowed_by_representative
                        for point in allowed_points
                    },
                    set(base.raw_points),
                )

    def test_unknown_and_work_allowlists_cover_the_declared_closure(self):
        fixture = tiny_fixture()

        for budget_type in (BudgetType.EQUAL_UNKNOWNS, BudgetType.EQUAL_WORK):
            bases = build_comparison_bases(fixture, budget_type, 2, 17)
            for mode, base in bases.items():
                with self.subTest(budget_type=budget_type, mode=mode):
                    allowed_by_representative = tuple(
                        base.allowed_points(fixture, representative)
                        for representative in base.representatives
                    )
                    self.assertEqual(
                        tuple(
                            point
                            for allowed_points in allowed_by_representative
                            for point in allowed_points
                        ),
                        base.raw_points,
                    )
                self.assertTrue(
                    all(
                        point in base.raw_points
                        for allowed_points in allowed_by_representative
                        for point in allowed_points
                    )
                )

    def test_equal_work_uses_exact_representative_count_and_common_limit(self):
        bases = build_comparison_bases(tiny_fixture(), BudgetType.EQUAL_WORK, 2, 41)

        self.assertEqual({base.effective_columns for base in bases.values()}, {2})
        self.assertEqual({len(base.representatives) for base in bases.values()}, {2})
        self.assertEqual({base.candidate_limit for base in bases.values()}, {2})

    def test_non_work_budgets_leave_candidate_limit_unapplied(self):
        fixture = tiny_fixture()

        for budget_type in (BudgetType.EQUAL_RAW, BudgetType.EQUAL_UNKNOWNS):
            with self.subTest(budget_type=budget_type):
                bases = build_comparison_bases(fixture, budget_type, 2, 41)
                self.assertEqual(
                    {base.candidate_limit for base in bases.values()},
                    {0},
                )

    def test_repeated_calls_are_deterministic_and_seed_changes_selection(self):
        fixture = tiny_fixture()
        first = build_comparison_bases(fixture, BudgetType.EQUAL_RAW, 4, 99)
        repeated = build_comparison_bases(fixture, BudgetType.EQUAL_RAW, 4, 99)
        other_seed = build_comparison_bases(fixture, BudgetType.EQUAL_RAW, 4, 100)

        self.assertEqual(first, repeated)
        self.assertNotEqual(first[Mode.CLASSIC].raw_points, other_seed[Mode.CLASSIC].raw_points)

    def test_invalid_budgets_and_impossible_requests_are_rejected(self):
        fixture = tiny_fixture()

        for budget_type in ("equal-raw", None, object()):
            with self.subTest(budget_type=budget_type):
                with self.assertRaises((TypeError, ValueError)):
                    build_comparison_bases(fixture, budget_type, 1, 17)
        for budget in (True, 1.0, 0, -1):
            with self.subTest(budget=budget):
                with self.assertRaises((TypeError, ValueError)):
                    build_comparison_bases(
                        fixture, BudgetType.EQUAL_UNKNOWNS, budget, 17
                    )
        impossible = (
            (BudgetType.EQUAL_RAW, fixture.n),
            (BudgetType.EQUAL_UNKNOWNS, 3),
            (BudgetType.EQUAL_WORK, 3),
        )
        for budget_type, budget in impossible:
            with self.subTest(budget_type=budget_type, budget=budget):
                with self.assertRaises(ValueError):
                    build_comparison_bases(fixture, budget_type, budget, 17)

    def test_invalid_fixtures_are_rejected(self):
        fixture = tiny_fixture()
        invalid_fixtures = (
            replace(fixture, generator=None),
            replace(fixture, generator=(0, 0)),
        )

        for invalid_fixture in invalid_fixtures:
            with self.subTest(fixture=invalid_fixture):
                with self.assertRaises(ValueError):
                    ranked_subgroup_points(invalid_fixture, 17)
                with self.assertRaises(ValueError):
                    build_comparison_bases(
                        invalid_fixture, BudgetType.EQUAL_RAW, 1, 17
                    )


class GeneratedFixtureTests(unittest.TestCase):
    def test_generated_fixture_builds_valid_repeatable_quotient_bases(self):
        fixture = generate_fixture(7, 20260804)
        first = build_comparison_bases(
            fixture, BudgetType.EQUAL_UNKNOWNS, 2, 20260804
        )
        second = build_comparison_bases(
            fixture, BudgetType.EQUAL_UNKNOWNS, 2, 20260804
        )

        self.assertEqual(first, second)
        for mode, base in first.items():
            with self.subTest(mode=mode):
                self.assertEqual(len(base.representatives), 2)
                self.assertEqual(len(set(base.representatives)), 2)
                self.assertNotIn(None, base.raw_points)
                self.assertTrue(
                    all(fixture.curve.is_on_curve(point) for point in base.raw_points)
                )
                self.assertTrue(
                    all(
                        representative_for(fixture, mode, point)
                        in base.representatives
                        for point in base.raw_points
                    )
                )


if __name__ == "__main__":
    unittest.main()
