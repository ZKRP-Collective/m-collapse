import ast
import math
import sys
import unittest
from dataclasses import FrozenInstanceError, fields
from pathlib import Path

from mcollapse.costs import CostVector
from mcollapse.curves import tiny_fixture
from mcollapse.factor_base import FactorBase, Mode
from mcollapse.metrics import (
    CensoredMetricSummary,
    CorrelationAssociation,
    CorrelationInterval,
    EffectSummary,
    FixtureObservation,
    FixtureSearchMetrics,
    HypothesisEvaluation,
    MIN_DEFINED_BOOTSTRAP_FRACTION,
    MULTIPLICITY_STATEMENT,
    REGISTERED_METRIC_FAMILIES,
    aggregate_fixture_search,
    average_ranks,
    clustered_interval,
    clustered_spearman,
    correlation_report,
    evaluate_hypothesis,
    holm_adjust,
    pearson,
    percentile,
    spearman,
    summarize_full_rank_work,
    verdict,
)
from mcollapse.relations import prepare_search


class PercentileTests(unittest.TestCase):
    def test_edges_and_linear_interpolation_are_deterministic(self):
        values = [40, 10, 30, 20]
        self.assertEqual(percentile(values, 0.0), 10.0)
        self.assertEqual(percentile(values, 1.0), 40.0)
        self.assertEqual(percentile(values, 0.25), 17.5)
        self.assertEqual(percentile([7], 0.37), 7.0)

    def test_rejects_empty_non_numeric_nonfinite_and_invalid_probability(self):
        cases = (
            ([], 0.5, ValueError),
            ([1, True], 0.5, TypeError),
            ([1, "2"], 0.5, TypeError),
            ([1, math.nan], 0.5, ValueError),
            ([1, math.inf], 0.5, ValueError),
            ([1], -0.01, ValueError),
            ([1], 1.01, ValueError),
            ([1], math.nan, ValueError),
            ([1], True, TypeError),
        )
        for values, probability, error in cases:
            with self.subTest(values=values, probability=probability):
                with self.assertRaises(error):
                    percentile(values, probability)


class ClusteredIntervalTests(unittest.TestCase):
    def test_is_deterministic_and_resamples_fixtures_not_targets(self):
        balanced = {"small": [0.0], "large": [10.0]}
        uneven = {"small": [0.0] * 100, "large": [10.0]}

        expected = clustered_interval(balanced, seed=19, iterations=1_000)
        self.assertEqual(expected, clustered_interval(balanced, seed=19, iterations=1_000))
        self.assertEqual(expected, clustered_interval(uneven, seed=19, iterations=1_000))
        self.assertEqual(expected, (0.0, 10.0))

    def test_statistic_receives_one_summary_per_sampled_fixture(self):
        interval = clustered_interval(
            {"a": [1.0, 3.0], "b": [8.0]},
            seed=2,
            iterations=100,
            statistic=max,
        )
        self.assertEqual(interval, (2.0, 8.0))

    def test_rejects_invalid_clusters_iterations_and_statistics(self):
        cases = (
            ({"only": [1.0]}, 10, ValueError),
            ({"a": [], "b": [1.0]}, 10, ValueError),
            ({"a": [1.0], "b": [math.inf]}, 10, ValueError),
            ({"a": [1.0], "b": [2.0]}, 0, ValueError),
            ({"a": [1.0], "b": [2.0]}, True, TypeError),
        )
        for clusters, iterations, error in cases:
            with self.subTest(clusters=clusters, iterations=iterations):
                with self.assertRaises(error):
                    clustered_interval(clusters, seed=1, iterations=iterations)
        with self.assertRaises(TypeError):
            clustered_interval({"a": [1.0], "b": [2.0]}, 1, 10, statistic=3)
        with self.assertRaises(ValueError):
            clustered_interval(
                {"a": [1.0], "b": [2.0]},
                1,
                10,
                statistic=lambda values: math.nan,
            )

    def test_fixture_ids_are_stable_ints_or_strings_only(self):
        class DerivedInt(int):
            pass

        class DerivedString(str):
            pass

        unstable = object()
        for fixture_id in (
            True,
            1.5,
            ("fixture", 1),
            unstable,
            DerivedInt(1),
            DerivedString("fixture"),
        ):
            with self.subTest(fixture_id=type(fixture_id).__name__):
                with self.assertRaises(TypeError):
                    clustered_interval({fixture_id: [1.0], "other": [2.0]}, 1, 10)

    def test_mixed_integer_and_string_fixture_ids_have_stable_order(self):
        first = {2: [1.0], "10": [2.0], 1: [3.0], "2": [4.0]}
        reversed_order = dict(reversed(tuple(first.items())))
        self.assertEqual(
            clustered_interval(first, seed=4, iterations=200),
            clustered_interval(reversed_order, seed=4, iterations=200),
        )


class CorrelationTests(unittest.TestCase):
    def test_average_ranks_ties_and_pearson(self):
        self.assertEqual(average_ranks([4, 4, 9, 9]), (1.5, 1.5, 3.5, 3.5))
        self.assertEqual(pearson([1, 2, 3], [20, 30, 40]), 1.0)
        self.assertEqual(pearson([1, 2, 3], [40, 30, 20]), -1.0)

    def test_spearman_has_exact_monotone_behavior_and_tie_handling(self):
        self.assertEqual(spearman([1, 2, 3, 4], [10, 20, 30, 40]), 1.0)
        self.assertEqual(spearman([1, 2, 3, 4], [40, 30, 20, 10]), -1.0)
        self.assertAlmostEqual(spearman([1, 1, 2, 3], [1, 2, 2, 4]), 5 / 6)

    def test_correlation_helpers_reject_constants_shapes_and_nonfinite_values(self):
        for function in (pearson, spearman):
            for left, right, error in (
                ([1], [2], ValueError),
                ([1, 2], [1], ValueError),
                ([1, 1], [2, 3], ValueError),
                ([1, 2], [3, 3], ValueError),
                ([1, math.nan], [2, 3], ValueError),
                ([1, True], [2, 3], TypeError),
            ):
                with self.subTest(function=function.__name__, left=left, right=right):
                    with self.assertRaises(error):
                        function(left, right)

    def test_clustered_spearman_is_deterministic_and_counts_undefined_resamples(self):
        records = {
            "f1": (1.0, 10.0),
            "f2": (2.0, 20.0),
            "f3": (3.0, 30.0),
            "f4": (4.0, 40.0),
        }
        result = clustered_spearman(records, seed=7, iterations=500)

        self.assertEqual(result, clustered_spearman(records, seed=7, iterations=500))
        self.assertEqual(result.coefficient, 1.0)
        self.assertEqual(result.interval, (1.0, 1.0))
        self.assertEqual(result.sample_size, 4)
        self.assertGreater(result.censored_resamples, 0)
        self.assertEqual(result.bootstrap_samples, 500)
        self.assertLessEqual(result.censored_resamples, result.bootstrap_samples)
        self.assertEqual(
            result.defined_resamples + result.censored_resamples,
            result.bootstrap_samples,
        )

    def test_clustered_spearman_allows_percentile_interval_to_exclude_estimate(self):
        result = clustered_spearman(
            {"f0": (0, 2), "f1": (1, 4), "f2": (2, 4)},
            seed=4,
            iterations=200,
        )

        self.assertLess(result.coefficient, result.interval[0])
        self.assertEqual(result.sample_size, 3)
        self.assertEqual(result.bootstrap_samples, 200)
        self.assertGreater(result.defined_resamples, 0)
        self.assertEqual(
            result.defined_resamples + result.censored_resamples,
            result.bootstrap_samples,
        )

    def test_clustered_spearman_rejects_target_rows_and_undefined_full_sample(self):
        with self.assertRaises(ValueError):
            clustered_spearman({"f1": [(1.0, 2.0), (2.0, 3.0)], "f2": (3.0, 4.0)}, 1, 10)
        with self.assertRaises(ValueError):
            clustered_spearman({"f1": (1.0, 2.0), "f2": (1.0, 3.0)}, 1, 10)

    def test_clustered_spearman_rejects_unstable_fixture_ids_before_repr_sorting(self):
        class AddressRepr:
            pass

        unstable = AddressRepr()
        for fixture_id in (True, 1.5, ("fixture", 1), unstable):
            with self.subTest(fixture_id=type(fixture_id).__name__):
                with self.assertRaises(TypeError):
                    clustered_spearman(
                        {fixture_id: (1.0, 2.0), "other": (2.0, 3.0)}, 1, 10
                    )

    def test_clustered_spearman_mixed_fixture_ids_are_deterministic(self):
        first = {2: (1.0, 1.0), "10": (2.0, 4.0), 1: (3.0, 9.0), "2": (4.0, 16.0)}
        reversed_order = dict(reversed(tuple(first.items())))
        self.assertEqual(
            clustered_spearman(first, seed=5, iterations=200),
            clustered_spearman(reversed_order, seed=5, iterations=200),
        )

    def test_report_contains_only_the_three_preregistered_fixture_associations(self):
        observations = tuple(
            FixtureObservation(
                f"fixture-{index}",
                orbit_compression=float(index),
                independent_relation_yield=float(index * 2),
                normalized_representation_multiplicity=float(index + 3),
                rank_increment_fraction=index / 10,
                calibrated_cost=float(20 - index),
            )
            for index in range(1, 6)
        )
        report = correlation_report(observations, seed=3, iterations=200)

        self.assertEqual(
            tuple(item.association for item in report),
            (
                "orbit compression vs independent-relation yield",
                "normalized representation multiplicity vs rank-increment fraction",
                "calibrated cost vs independent-relation yield",
            ),
        )
        self.assertTrue(all("caus" not in item.association for item in report))
        self.assertTrue(all(item.result.sample_size == 5 for item in report))

    def test_correlation_interpretability_uses_exact_95_percent_boundary(self):
        at_boundary = CorrelationInterval(0.5, (0.2, 0.8), 4, 100, 5)
        below_boundary = CorrelationInterval(0.5, (0.2, 0.8), 4, 100, 6)
        negative = CorrelationInterval(-0.5, (-0.8, -0.2), 4, 100, 0)
        crossing = CorrelationInterval(0.0, (-0.2, 0.2), 4, 100, 0)
        touching_below = CorrelationInterval(-0.1, (-0.2, 0.0), 4, 100, 0)
        touching_above = CorrelationInterval(0.1, (0.0, 0.2), 4, 100, 0)

        self.assertEqual(MIN_DEFINED_BOOTSTRAP_FRACTION, 0.95)
        self.assertTrue(at_boundary.interpretable)
        self.assertEqual(at_boundary.status, "positive")
        self.assertEqual(negative.status, "negative")
        self.assertEqual(crossing.status, "unresolved")
        self.assertEqual(touching_below.status, "unresolved")
        self.assertEqual(touching_above.status, "unresolved")
        self.assertFalse(below_boundary.interpretable)
        self.assertEqual(below_boundary.status, "unresolved")
        self.assertEqual(
            {item.status for item in (at_boundary, negative, crossing, touching_below)},
            {"positive", "negative", "unresolved"},
        )

    def test_degenerate_correlation_report_is_unresolved_despite_positive_interval(self):
        observations = tuple(
            FixtureObservation(
                f"fixture-{index}",
                orbit_compression=float(index == 19),
                independent_relation_yield=float(index == 19),
                normalized_representation_multiplicity=float(index == 19),
                rank_increment_fraction=float(index == 19),
                calibrated_cost=float(index == 19),
            )
            for index in range(20)
        )
        report = correlation_report(observations, seed=7, iterations=1_000)

        self.assertTrue(all(item.result.interval == (1.0, 1.0) for item in report))
        self.assertTrue(all(300 < item.result.censored_resamples < 400 for item in report))
        self.assertTrue(all(item.status == "unresolved" for item in report))
        self.assertTrue(
            all(item.status not in ("positive", "negative", "decisive") for item in report)
        )


class HolmTests(unittest.TestCase):
    def test_registered_metric_families_are_exact_and_immutable(self):
        self.assertEqual(
            dict(REGISTERED_METRIC_FAMILIES),
            {
                "H1": (
                    "independent-relations-per-unknown",
                    "independent-relations-per-work",
                    "work-to-full-rank",
                    "nullity-after-equal-candidates",
                ),
                "H2": (
                    "decomposition-probability",
                    "unique-normalized-vectors-per-target",
                    "rank-increment-fraction",
                    "conditional-representation-multiplicity",
                ),
                "H3": (
                    "calibrated-cost-per-independent-relation",
                    "primitive-cost-per-accepted-relation",
                    "primitive-cost-per-independent-relation",
                    "warm-wall-time-per-independent-relation",
                ),
            },
        )
        with self.assertRaises(TypeError):
            REGISTERED_METRIC_FAMILIES["H1"] = ()

    def test_known_vector_and_original_order_restoration(self):
        self.assertEqual(holm_adjust([0.01, 0.04, 0.03, 0.005]), (0.03, 0.06, 0.06, 0.02))
        self.assertEqual(holm_adjust([0.04, 0.001, 0.02]), (0.04, 0.003, 0.04))

    def test_rejects_values_outside_closed_unit_interval(self):
        for values, error in (
            ([True], TypeError),
            (["0.1"], TypeError),
            ([math.nan], ValueError),
            ([math.inf], ValueError),
            ([-0.01], ValueError),
            ([1.01], ValueError),
        ):
            with self.subTest(values=values):
                with self.assertRaises(error):
                    holm_adjust(values)


class VerdictTests(unittest.TestCase):
    def test_h1_and_h2_support_boundaries_and_band_direction(self):
        for name in ("H1", "H2"):
            self.assertEqual(
                verdict(name, 1.10, (1.0001, 1.3), [1.01, 1.02, 1.03]),
                "supported in tested regime",
            )
            for estimate, interval, bands in (
                (1.0999, (1.01, 1.3), [1.1, 1.1, 1.1]),
                (1.10, (1.0, 1.3), [1.1, 1.1, 1.1]),
                (1.10, (1.01, 1.3), [1.1, 1.0, 1.1]),
            ):
                with self.subTest(name=name, estimate=estimate, interval=interval, bands=bands):
                    self.assertEqual(verdict(name, estimate, interval, bands), "inconclusive")

    def test_h1_and_h2_reject_from_interval_regardless_of_mixed_bands(self):
        for name in ("H1", "H2"):
            self.assertEqual(
                verdict(name, 0.9, (0.8, 1.0), [1.0, 0.9, 0.8]),
                "rejected in tested regime",
            )
            self.assertEqual(
                verdict(name, 0.9, (0.8, 1.0001), [0.9, 0.9, 0.9]),
                "inconclusive",
            )
            self.assertEqual(
                verdict(name, 0.9, (0.8, 1.0), [0.9, 1.01, 0.9]),
                "rejected in tested regime",
            )

    def test_h3_strict_support_boundaries_and_band_direction(self):
        self.assertEqual(
            verdict("H3", 0.90, (0.7, 0.8999), [0.9, 0.8, 0.7]),
            "supported in tested regime",
        )
        for estimate, interval, bands in (
            (0.9001, (0.7, 0.89), [0.8, 0.8, 0.8]),
            (0.90, (0.7, 0.90), [0.8, 0.8, 0.8]),
            (0.90, (0.7, 0.89), [0.8, 1.0, 0.8]),
        ):
            with self.subTest(estimate=estimate, interval=interval, bands=bands):
                self.assertEqual(verdict("H3", estimate, interval, bands), "inconclusive")

    def test_h3_rejects_from_interval_regardless_of_mixed_bands(self):
        self.assertEqual(
            verdict("H3", 1.1, (1.0, 1.3), [1.0, 1.1, 1.2]),
            "rejected in tested regime",
        )
        self.assertEqual(
            verdict("H3", 1.1, (0.9999, 1.3), [1.1, 1.1, 1.1]),
            "inconclusive",
        )
        self.assertEqual(
            verdict("H3", 1.1, (1.0, 1.3), [1.1, 0.99, 1.1]),
            "rejected in tested regime",
        )

    def test_verdict_rejects_unknown_names_bad_intervals_bands_and_nonfinite_values(self):
        cases = (
            ("H4", 1.1, (1.0, 1.2), [1.1, 1.1, 1.1]),
            ("H1", 1.1, (1.2, 1.0), [1.1, 1.1, 1.1]),
            ("H1", math.nan, (1.0, 1.2), [1.1, 1.1, 1.1]),
            ("H1", 1.1, (1.0, math.inf), [1.1, 1.1, 1.1]),
            ("H1", 1.1, (1.0, 1.2), [1.1, 1.1]),
            ("H1", 1.1, (1.0, 1.2), [1.1, 1.1, math.nan]),
        )
        for arguments in cases:
            with self.subTest(arguments=arguments):
                with self.assertRaises(ValueError):
                    verdict(*arguments)

    def test_holm_gates_primary_support_and_marks_supporting_secondary_metrics(self):
        summary = EffectSummary(1.2, (1.1, 1.3), (1.1, 1.2, 1.3))
        family = REGISTERED_METRIC_FAMILIES["H2"]
        report = evaluate_hypothesis(
            "H2",
            summary,
            metric_p_values={
                family[0]: 0.01,
                family[1]: 0.01,
                family[2]: 0.04,
                family[3]: 0.001,
            },
            direction_supporting_metrics=(family[1], family[2]),
        )

        self.assertEqual(report.verdict, "supported in tested regime")
        self.assertEqual(report.primary_metric, family[0])
        self.assertEqual(report.raw_primary_p_value, 0.01)
        self.assertEqual(report.adjusted_primary_p_value, 0.03)
        self.assertEqual(
            dict(report.adjusted_p_values),
            {family[0]: 0.03, family[1]: 0.03, family[2]: 0.04, family[3]: 0.004},
        )
        self.assertEqual(report.supporting_secondary_metrics, (family[1], family[2]))
        self.assertEqual(report.multiplicity_statement, MULTIPLICITY_STATEMENT)
        self.assertIn("Multiplicity-adjusted support decision", report.multiplicity_statement)
        self.assertIn("support decision", report.multiplicity_statement)
        self.assertIn("interval is unadjusted", report.multiplicity_statement)

    def test_primary_holm_failure_makes_effect_support_inconclusive(self):
        summary = EffectSummary(1.2, (1.1, 1.3), (1.1, 1.2, 1.3))
        family = REGISTERED_METRIC_FAMILIES["H1"]
        report = evaluate_hypothesis(
            "H1",
            summary,
            metric_p_values={
                family[0]: 0.02,
                family[1]: 0.5,
                family[2]: 0.6,
                family[3]: 0.7,
            },
        )

        self.assertEqual(report.raw_primary_p_value, 0.02)
        self.assertEqual(report.adjusted_primary_p_value, 0.08)
        self.assertEqual(report.verdict, "inconclusive")

    def test_primary_holm_support_boundary_is_inclusive_at_point_zero_five(self):
        summary = EffectSummary(1.2, (1.1, 1.3), (1.1, 1.2, 1.3))
        family = REGISTERED_METRIC_FAMILIES["H2"]
        supported = evaluate_hypothesis(
            "H2",
            summary,
            metric_p_values={family[0]: 0.0125, family[1]: 0.9, family[2]: 0.9, family[3]: 0.9},
        )
        inconclusive = evaluate_hypothesis(
            "H2",
            summary,
            metric_p_values={family[0]: 0.0126, family[1]: 0.9, family[2]: 0.9, family[3]: 0.9},
        )

        self.assertEqual(supported.adjusted_primary_p_value, 0.05)
        self.assertEqual(supported.verdict, "supported in tested regime")
        self.assertGreater(inconclusive.adjusted_primary_p_value, 0.05)
        self.assertEqual(inconclusive.verdict, "inconclusive")

    def test_full_registered_family_controls_primary_support(self):
        summary = EffectSummary(1.2, (1.1, 1.3), (1.1, 1.2, 1.3))
        family = REGISTERED_METRIC_FAMILIES["H1"]
        with self.assertRaises(ValueError):
            evaluate_hypothesis("H1", summary, metric_p_values={family[0]: 0.04})

        adjusted_out = evaluate_hypothesis(
            "H1", summary, metric_p_values={metric: 0.04 for metric in family}
        )
        supported = evaluate_hypothesis(
            "H1", summary, metric_p_values={metric: 0.001 for metric in family}
        )

        self.assertEqual(adjusted_out.adjusted_primary_p_value, 0.16)
        self.assertEqual(adjusted_out.verdict, "inconclusive")
        self.assertEqual(supported.adjusted_primary_p_value, 0.004)
        self.assertEqual(supported.verdict, "supported in tested regime")


class PublicModelValidationTests(unittest.TestCase):
    def test_effect_summary_allows_percentile_interval_to_exclude_estimate(self):
        summary = EffectSummary(1.3, (1.0, 1.2), (1.1, 1.1, 1.1))
        self.assertEqual(summary.estimate, 1.3)
        self.assertEqual(summary.interval, (1.0, 1.2))

    def test_effect_summary_requires_finite_estimate_and_ordered_interval(self):
        for arguments, error in (
            ((math.nan, (1.0, 1.2), (1.1, 1.1, 1.1)), ValueError),
            ((1.1, (1.2, 1.0), (1.1, 1.1, 1.1)), ValueError),
            ((1.1, (1.0, math.inf), (1.1, 1.1, 1.1)), ValueError),
        ):
            with self.subTest(arguments=arguments):
                with self.assertRaises(error):
                    EffectSummary(*arguments)

    def test_correlation_interval_allows_percentile_interval_to_exclude_estimate(self):
        valid = CorrelationInterval(0.9, (0.2, 0.8), 4, 100, 7)
        self.assertEqual(valid.interval, (0.2, 0.8))
        self.assertEqual(valid.defined_resamples, 93)

    def test_correlation_interval_validates_finite_values_and_count_relationships(self):
        for arguments, error in (
            ((math.nan, (0.2, 0.8), 4, 100, 7), ValueError),
            ((0.5, (0.8, 0.2), 4, 100, 7), ValueError),
            ((0.5, (0.2, math.inf), 4, 100, 7), ValueError),
            ((0.5, [0.2, 0.8], 4, 100, 7), TypeError),
            ((0.5, (0.2, 0.8), True, 100, 7), TypeError),
            ((0.5, (0.2, 0.8), 1, 100, 7), ValueError),
            ((0.5, (0.2, 0.8), 4, 0, 0), ValueError),
            ((0.5, (0.2, 0.8), 4, -1, 7), ValueError),
            ((0.5, (0.2, 0.8), 4, 100, -1), ValueError),
            ((0.5, (0.2, 0.8), 4, 100, 100), ValueError),
            ((0.5, (0.2, 0.8), 4, 100, 101), ValueError),
        ):
            with self.subTest(arguments=arguments):
                with self.assertRaises(error):
                    CorrelationInterval(*arguments)

    def test_correlation_association_rejects_invalid_or_causal_labels(self):
        result = CorrelationInterval(0.5, (0.2, 0.8), 4, 100, 7)
        for association, error in (
            ("", ValueError),
            ("cost causes yield", ValueError),
            ("causal effect of cost", ValueError),
            ("cost-yield", ValueError),
            (3, TypeError),
        ):
            with self.subTest(association=association):
                with self.assertRaises(error):
                    CorrelationAssociation(association, result)
        with self.assertRaises(TypeError):
            CorrelationAssociation("cost vs yield", object())

    def test_hypothesis_evaluation_validates_verdict_and_immutable_sequences(self):
        valid = (
            "inconclusive",
            "primary",
            0.01,
            0.02,
            (("primary", 0.02),),
            (),
            MULTIPLICITY_STATEMENT,
        )
        for arguments, error in (
            (("unknown", *valid[1:]), ValueError),
            ((valid[0], "", *valid[2:]), ValueError),
            ((valid[0], valid[1], math.nan, *valid[3:]), ValueError),
            ((valid[0], valid[1], valid[2], 1.01, *valid[4:]), ValueError),
            ((valid[0], valid[1], 0.02, 0.01, (("primary", 0.01),), *valid[5:]), ValueError),
            ((valid[0], valid[1], valid[2], valid[3], [("primary", 0.02)], *valid[5:]), TypeError),
            ((valid[0], valid[1], valid[2], valid[3], (("primary", math.nan),), *valid[5:]), ValueError),
            ((valid[0], valid[1], valid[2], valid[3], (("other", 0.02),), *valid[5:]), ValueError),
            ((valid[0], valid[1], valid[2], valid[3], valid[4], ["yield"], valid[6]), TypeError),
            ((valid[0], valid[1], valid[2], valid[3], valid[4], ("",), valid[6]), ValueError),
            (("supported in tested regime", valid[1], 0.02, 0.06, (("primary", 0.06),), (), valid[6]), ValueError),
            ((valid[0], valid[1], 0.01, 0.02, (("primary", 0.02), ("yield", 0.06)), ("yield",), valid[6]), ValueError),
            ((valid[0], valid[1], valid[2], valid[3], valid[4], (), "adjusted interval"), ValueError),
        ):
            with self.subTest(arguments=arguments):
                with self.assertRaises(error):
                    HypothesisEvaluation(*arguments)

    def test_evaluation_requires_exact_registered_family(self):
        summary = EffectSummary(1.2, (1.1, 1.3), (1.1, 1.2, 1.3))
        family = REGISTERED_METRIC_FAMILIES["H1"]
        complete = {metric: 0.01 for metric in family}
        for p_values, supporting, error in (
            ({}, (), ValueError),
            ({family[0]: 0.01}, (), ValueError),
            ({metric: 0.01 for metric in family[:-1]}, (), ValueError),
            ({**complete, "unexpected": 0.01}, (), ValueError),
            (complete, ("missing",), ValueError),
            ([(family[0], 0.01)], (), TypeError),
        ):
            with self.subTest(p_values=p_values):
                with self.assertRaises(error):
                    evaluate_hypothesis(
                        "H1",
                        summary,
                        metric_p_values=p_values,
                        direction_supporting_metrics=supporting,
                    )

    def test_fixture_search_metrics_validates_cost_counts_and_calibration(self):
        for arguments, error in (
            ((object(), 1, 1, 1), TypeError),
            ((CostVector(), True, 1, 1), TypeError),
            ((CostVector(), -1, 1, 1), ValueError),
            ((CostVector(), 1, -1, 1), ValueError),
            ((CostVector(), 1, 1, -1), ValueError),
            ((CostVector(), 1, 1, 1, math.nan), ValueError),
            ((CostVector(), 1, 1, 1, -0.1), ValueError),
        ):
            with self.subTest(arguments=arguments):
                with self.assertRaises(error):
                    FixtureSearchMetrics(*arguments)

    def test_result_count_models_do_not_invent_cross_count_orderings(self):
        search = FixtureSearchMetrics(CostVector(), 0, 1, 3)
        censored = CensoredMetricSummary((1.0,), 1, 5)

        self.assertEqual(search.accepted_relations, 3)
        self.assertEqual(censored.censored_count, 5)

    def test_censored_summary_validates_values_and_exact_counts(self):
        for arguments, error in (
            (([1.0], 1, 0), TypeError),
            (((math.inf,), 1, 0), ValueError),
            (((0.0,), 1, 0), ValueError),
            (((1.0,), True, 0), TypeError),
            (((1.0,), 2, 0), ValueError),
            (((1.0,), 1, -1), ValueError),
        ):
            with self.subTest(arguments=arguments):
                with self.assertRaises(error):
                    CensoredMetricSummary(*arguments)


class FixtureAggregationTests(unittest.TestCase):
    def test_models_are_immutable_and_validate_fixture_level_values(self):
        observation = FixtureObservation("fixture", 2, 3, 4, 0.5, 6, None)
        with self.assertRaises(FrozenInstanceError):
            observation.calibrated_cost = 0
        for arguments in (
            ("", 2, 3, 4, 0.5, 6, None),
            ("fixture", -1, 3, 4, 0.5, 6, None),
            ("fixture", 2, 3, 4, 1.1, 6, None),
            ("fixture", 2, 3, 4, 0.5, 6, 0.0),
            ("fixture", 2, 3, 4, 0.5, math.inf, None),
        ):
            with self.subTest(arguments=arguments):
                with self.assertRaises(ValueError):
                    FixtureObservation(*arguments)
        for fixture_id in (True, 1.5, ("fixture", 1), object()):
            with self.subTest(fixture_id=type(fixture_id).__name__):
                with self.assertRaises(TypeError):
                    FixtureObservation(fixture_id, 2, 3, 4, 0.5, 6, None)

    def test_100_target_aggregation_charges_preparation_once(self):
        fixture = tiny_fixture()
        point = fixture.generator
        base = FactorBase(Mode.CLASSIC, (point,), (point,), 1, 0)
        prepared = prepare_search(fixture, base, 2, 4)
        results = tuple(
            prepared.search(fixture.curve.mul(index % (fixture.n - 1) + 1, point))
            for index in range(100)
        )
        metrics = aggregate_fixture_search(prepared, results)
        expected = prepared.preparation_cost + sum(
            (result.search_cost for result in results), CostVector()
        )

        self.assertEqual(metrics.total_cost, expected)
        self.assertEqual(
            metrics.total_candidates,
            prepared.preparation_candidates + sum(result.search_candidates for result in results),
        )
        self.assertEqual(metrics.target_count, 100)
        self.assertEqual(metrics.accepted_relations, sum(len(result.relations) for result in results))
        self.assertNotEqual(
            metrics.total_cost,
            prepared.preparation_cost * 100
            + sum((result.search_cost for result in results), CostVector()),
        )

    def test_censored_full_rank_work_is_counted_and_never_replaced_with_zero(self):
        observations = (
            FixtureObservation("a", 2, 3, 4, 0.5, 6, 12),
            FixtureObservation("b", 2, 3, 4, 0.5, 6, None),
            FixtureObservation("c", 2, 3, 4, 0.5, 6, 18),
        )
        summary = summarize_full_rank_work(observations)

        self.assertEqual(summary.values, (12.0, 18.0))
        self.assertEqual(summary.observed_count, 2)
        self.assertEqual(summary.censored_count, 1)
        self.assertNotIn(0.0, summary.values)

    def test_task6_uses_only_the_standard_library(self):
        source = Path(__file__).parents[1].joinpath("metrics.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        imports = {
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, ast.Import)
            for alias in node.names
        }
        imports.update(
            node.module
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.module
        )
        unexpected = {
            module
            for module in imports
            if module != "__future__"
            and not module.startswith("mcollapse.")
            and module.split(".")[0] not in sys.stdlib_module_names
        }
        self.assertEqual(unexpected, set())
        self.assertIn("mcollapse.costs", imports)
        self.assertIn("mcollapse.relations", imports)
        self.assertTrue(all(field.name for field in fields(FixtureObservation)))


if __name__ == "__main__":
    unittest.main()
