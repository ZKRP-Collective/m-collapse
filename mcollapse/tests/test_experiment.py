from __future__ import annotations

import ast
import hashlib
import inspect
import json
import os
import tempfile
import unittest
from dataclasses import FrozenInstanceError
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from mcollapse.costs import CostVector
from mcollapse.curves import generate_fixture, tiny_fixture
from mcollapse.experiment import (
    CellKey,
    Manifest,
    atomic_json,
    load_manifest,
    run_cell,
    run_manifest,
)
from mcollapse.factor_base import BudgetType, Mode
from mcollapse.metrics import REGISTERED_METRIC_FAMILIES
from mcollapse.relations import PreparedSearch, Relation, SearchResult
from mcollapse.run_experiment import main


ROOT = Path(__file__).resolve().parents[2]
COST_NAMES = tuple(CostVector.__dataclass_fields__)
CALIBRATION = {
    "fp_mul": 1.0,
    "fp_square": 0.8,
    "fp_inv": 80.0,
    "point_add": 0.0,
    "point_double": 0.0,
    "endomorphism": 0.0,
    "table_probe": 0.1,
    "normalization": 0.05,
}


def mini_manifest(**changes: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema": 1,
        "name": "test-mini",
        "subgroup_bits": [6],
        "fixture_seeds": [1],
        "factor_base_budgets": [2],
        "arities": [2],
        "budget_types": ["equal-raw"],
        "targets_per_cell": 3,
        "candidate_limit": 100,
        "bootstrap_iterations": 20,
        "bootstrap_seed": 20260804,
        "calibration_weights": dict(CALIBRATION),
    }
    payload.update(changes)
    return payload


def write_manifest(directory: Path, payload: dict[str, object]) -> Path:
    path = directory / "manifest.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class ManifestTests(unittest.TestCase):
    def test_manifest_schema_is_strict_and_immutable(self):
        manifest = Manifest.from_mapping(mini_manifest())
        self.assertEqual(manifest.subgroup_bits, (6,))
        self.assertEqual(tuple(manifest.calibration_weights), COST_NAMES)
        with self.assertRaises((FrozenInstanceError, AttributeError)):
            manifest.name = "changed"  # type: ignore[misc]
        with self.assertRaises(TypeError):
            manifest.calibration_weights["fp_mul"] = 2.0  # type: ignore[index]

        invalid_payloads = []
        missing = mini_manifest()
        del missing["name"]
        invalid_payloads.append(missing)
        invalid_payloads.append(mini_manifest(extra=True))
        invalid_payloads.append(mini_manifest(schema=True))
        invalid_payloads.append(mini_manifest(subgroup_bits="6"))
        invalid_payloads.append(mini_manifest(subgroup_bits=[6, 6]))
        invalid_payloads.append(mini_manifest(fixture_seeds=[1, 1]))
        invalid_payloads.append(mini_manifest(factor_base_budgets=[2, 2]))
        invalid_payloads.append(mini_manifest(arities=[2, 2]))
        invalid_payloads.append(mini_manifest(budget_types=["equal-raw", "equal-raw"]))
        invalid_payloads.append(mini_manifest(targets_per_cell=0))
        invalid_payloads.append(mini_manifest(calibration_weights={"fp_mul": 1.0}))
        for payload in invalid_payloads:
            with self.subTest(payload=payload), self.assertRaises((TypeError, ValueError)):
                Manifest.from_mapping(payload)

    def test_json_loader_rejects_duplicate_object_keys(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "duplicate.json"
            path.write_text('{"schema":1,"schema":1}', encoding="utf-8")
            with self.assertRaises(ValueError):
                load_manifest(path)

    def test_registered_manifests_have_exact_values(self):
        smoke = load_manifest(ROOT / "mcollapse/manifests/h1_h3_smoke.json")
        version_one = load_manifest(ROOT / "mcollapse/manifests/h1_h3_v1.json")

        self.assertEqual(smoke.subgroup_bits, (8, 10, 12))
        self.assertEqual(smoke.fixture_seeds, (1, 2))
        self.assertEqual(smoke.factor_base_budgets, (4, 8))
        self.assertEqual(smoke.arities, (2, 3))
        self.assertEqual(smoke.budget_types, tuple(BudgetType))
        self.assertEqual(smoke.targets_per_cell, 100)
        self.assertEqual(smoke.candidate_limit, 10_000)
        self.assertEqual(smoke.bootstrap_iterations, 2_000)
        self.assertEqual(smoke.bootstrap_seed, 20260804)

        self.assertEqual(version_one.subgroup_bits, (12, 14, 16))
        self.assertEqual(version_one.fixture_seeds, tuple(range(1, 21)))
        self.assertEqual(version_one.factor_base_budgets, (8, 16, 32))
        self.assertEqual(version_one.arities, (2, 3))
        self.assertEqual(version_one.budget_types, tuple(BudgetType))
        self.assertEqual(version_one.targets_per_cell, 10_000)
        self.assertEqual(version_one.candidate_limit, 10_000)
        self.assertEqual(version_one.bootstrap_iterations, 10_000)
        self.assertEqual(version_one.bootstrap_seed, 20260804)
        self.assertEqual(dict(smoke.calibration_weights), CALIBRATION)
        self.assertEqual(dict(version_one.calibration_weights), CALIBRATION)


class RunnerTests(unittest.TestCase):
    def test_two_independent_runs_have_identical_deterministic_results(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest_path = write_manifest(root, mini_manifest())
            first = run_manifest(manifest_path, root / "first")
            second = run_manifest(manifest_path, root / "second")
            self.assertEqual(first.manifest_hash, second.manifest_hash)
            self.assertEqual(first.completed_cells, second.completed_cells)
            self.assertEqual(dict(first.summary), dict(second.summary))
            for relative in ("manifest.json", "summary.json", "report.md"):
                self.assertEqual(
                    (root / "first" / relative).read_bytes(),
                    (root / "second" / relative).read_bytes(),
                )
            first_cell = next((root / "first/cells").glob("*.json"))
            second_cell = next((root / "second/cells").glob("*.json"))
            self.assertEqual(first_cell.read_bytes(), second_cell.read_bytes())

            science = json.loads(first_cell.read_text(encoding="utf-8"))
            self.assertNotIn("warm_wall_time_ns", science)
            timing_files = tuple((root / "first/diagnostics").glob("*.timing.json"))
            self.assertEqual(len(timing_files), 1)
            timing = json.loads(timing_files[0].read_text(encoding="utf-8"))
            for samples in timing["modes"].values():
                self.assertGreaterEqual(len(samples["total_ns_samples"]), 3)
                self.assertTrue(samples["machine_dependent"])

    def test_resume_reuses_final_cells_and_ignores_corrupt_temporary_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest_path = write_manifest(root, mini_manifest())
            output = root / "output"
            first = run_manifest(manifest_path, output)
            cell_path = next((output / "cells").glob("*.json"))
            before = (cell_path.stat().st_mtime_ns, file_sha256(cell_path))
            corrupt = cell_path.with_suffix(".json.tmp")
            corrupt.write_text("not json", encoding="utf-8")

            resumed = run_manifest(manifest_path, output, resume=True)

            self.assertEqual(first.completed_cells, resumed.completed_cells)
            self.assertEqual(before, (cell_path.stat().st_mtime_ns, file_sha256(cell_path)))
            self.assertEqual(corrupt.read_text(encoding="utf-8"), "not json")

    def test_resume_rejects_manifest_and_source_mismatches(self):
        for field in (
            "manifest_hash", "source_commit", "source_identity",
            "fixture_resolution_id",
        ):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                manifest_path = write_manifest(root, mini_manifest())
                output = root / "output"
                run_manifest(manifest_path, output)
                cell_path = next((output / "cells").glob("*.json"))
                payload = json.loads(cell_path.read_text(encoding="utf-8"))
                payload[field] = "mismatch"
                atomic_json(cell_path, payload)
                with self.assertRaisesRegex(ValueError, field.replace("_", " ")):
                    run_manifest(manifest_path, output, resume=True)

    def test_final_checkpoint_hash_and_deep_schema_reject_forgery(self):
        import mcollapse.experiment as experiment

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest_path = write_manifest(root, mini_manifest(targets_per_cell=1))
            output = root / "output"
            run_manifest(manifest_path, output)
            cell_path = next((output / "cells").glob("*.json"))
            payload = json.loads(cell_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["payload_hash"], experiment._payload_hash(payload))

            payload["modes"][Mode.CLASSIC.value]["totals"]["targets"] = -1
            payload["payload_hash"] = experiment._payload_hash(payload)
            atomic_json(cell_path, payload)
            with self.assertRaisesRegex(ValueError, "targets"):
                run_manifest(manifest_path, output, resume=True)

    def test_progress_hash_and_monotone_aggregate_index_reject_forgery(self):
        import mcollapse.experiment as experiment

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest_path = write_manifest(root, mini_manifest(targets_per_cell=2))
            output = root / "output"
            original_search = PreparedSearch.search
            calls = 0

            def interrupt(prepared, target):
                nonlocal calls
                calls += 1
                if calls == len(Mode) + 1:
                    raise RuntimeError("interrupt")
                return original_search(prepared, target)

            with (
                patch.object(experiment, "_CHECKPOINT_BATCH", 1),
                patch.object(experiment, "_measure_warm_timings", return_value={}),
                patch.object(PreparedSearch, "search", interrupt),
            ):
                with self.assertRaisesRegex(RuntimeError, "interrupt"):
                    run_manifest(manifest_path, output)

            progress_path = next((output / "cells/.progress").glob("*.json"))
            payload = json.loads(progress_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["payload_hash"], experiment._payload_hash(payload))
            self.assertEqual(payload["next_target_index"], 1)
            classic = payload["modes"][Mode.CLASSIC.value]
            classic["processed_targets"] = 0
            payload["payload_hash"] = experiment._payload_hash(payload)
            atomic_json(progress_path, payload)
            with self.assertRaisesRegex(ValueError, "monotone|target index"):
                run_manifest(manifest_path, output, resume=True)

    def test_relation_search_receives_only_public_api_values_and_no_oracle_data_is_serialized(self):
        import mcollapse.experiment as experiment

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest_path = write_manifest(root, mini_manifest())
            original_prepare = experiment.prepare_search
            calls: list[tuple[type[object], ...]] = []

            def checked_prepare(*args: object, **kwargs: object):
                calls.append(tuple(type(value) for value in args))
                self.assertEqual(kwargs, {})
                self.assertEqual(len(args), 4)
                return original_prepare(*args)

            with patch.object(experiment, "prepare_search", checked_prepare):
                run_manifest(manifest_path, root / "output")

            self.assertEqual(len(calls), len(Mode))
            serialized = "\n".join(
                path.read_text(encoding="utf-8")
                for path in (root / "output").rglob("*.json")
                if not path.name.endswith(".tmp")
            )
            for forbidden in ('"k"', "secret_scalar", "representative_logs", "log_map"):
                self.assertNotIn(forbidden, serialized)
            self.assertEqual(tuple(inspect.signature(original_prepare).parameters), (
                "fixture", "base", "arity", "candidate_limit"
            ))

    def test_actual_relation_control_rows_retain_complete_public_equation(self):
        manifest = Manifest.from_mapping(mini_manifest(targets_per_cell=1))
        key = CellKey(6, 1, 2, 2, BudgetType.EQUAL_RAW)
        import mcollapse.experiment as experiment

        original = experiment._actual_relation_control_rows
        observed = []

        def inspect_rows(*args: object, **kwargs: object):
            expanded, compact, orbit_count = original(*args, **kwargs)
            target = args[4]
            suffix = ((-target["v"]) % args[0].fixture.n, target["u"] % args[0].fixture.n)
            for row in expanded:
                self.assertEqual(len(row), 3 * orbit_count + 2)
                self.assertEqual(row[-2:], suffix)
            for row in compact:
                self.assertEqual(len(row), orbit_count + 2)
                self.assertEqual(row[-2:], suffix)
            observed.extend(expanded)
            observed.extend(compact)
            return expanded, compact, orbit_count

        with patch.object(experiment, "_actual_relation_control_rows", inspect_rows):
            run_cell(manifest, key, "manifest-hash", "source-commit")
        self.assertTrue(observed)

    def test_actual_relation_public_suffix_perturbations_abort_exactly(self):
        manifest = Manifest.from_mapping(mini_manifest(targets_per_cell=1))
        key = CellKey(6, 1, 2, 2, BudgetType.EQUAL_RAW)
        import mcollapse.experiment as experiment

        original = experiment._actual_relation_control_rows
        for suffix_index in (-2, -1):
            with self.subTest(suffix_index=suffix_index):
                def perturbed(*args: object, **kwargs: object):
                    expanded, compact, orbit_count = original(*args, **kwargs)
                    if not compact:
                        return expanded, compact, orbit_count
                    changed = list(compact)
                    row = list(changed[0])
                    row[suffix_index] = (row[suffix_index] + 1) % args[0].fixture.n
                    changed[0] = tuple(row)
                    return expanded, tuple(changed), orbit_count

                with patch.object(experiment, "_actual_relation_control_rows", perturbed):
                    with self.assertRaises(ArithmeticError) as caught:
                        run_cell(manifest, key, "manifest-hash", "source-commit")
                self.assertEqual(str(caught.exception), "GLV/MCOLLAPSE row-space mismatch")

    def test_glv_control_explicitly_augments_with_constraint_rows(self):
        import mcollapse.experiment as experiment

        manifest = Manifest.from_mapping(mini_manifest(targets_per_cell=1))
        key = CellKey(6, 1, 2, 2, BudgetType.EQUAL_RAW)
        original = experiment.glv_constraint_rows
        calls: list[tuple[int, int, int, bool]] = []

        def capture(*args, **kwargs):
            calls.append((*args, kwargs.get("include_rhs", True)))
            return original(*args, **kwargs)

        with patch.object(experiment, "glv_constraint_rows", capture):
            result = run_cell(manifest, key, "manifest-hash", "source-commit")
        self.assertTrue(calls)
        self.assertTrue(all(call[3] is False for call in calls))
        self.assertGreater(result.glv_control_details["explicit_constraint_rows"], 0)

    def test_each_mode_is_prepared_once_and_preparation_is_charged_once(self):
        import mcollapse.experiment as experiment

        manifest = Manifest.from_mapping(mini_manifest(targets_per_cell=4))
        key = CellKey(6, 1, 2, 2, BudgetType.EQUAL_RAW)
        original_prepare = experiment.prepare_search
        original_aggregate = experiment.aggregate_fixture_search
        prepared_modes: list[Mode] = []
        aggregate_modes: list[Mode] = []

        def counted_prepare(*args: object, **kwargs: object):
            prepared = original_prepare(*args, **kwargs)
            prepared_modes.append(prepared.base.mode)
            return prepared

        def checked_aggregate(prepared, results, weights):
            materialized = tuple(results)
            aggregate_modes.append(prepared.base.mode)
            aggregate = original_aggregate(prepared, materialized, weights)
            expected_candidates = prepared.preparation_candidates + sum(
                result.search_candidates for result in materialized
            )
            self.assertEqual(aggregate.total_candidates, expected_candidates)
            return aggregate

        with (
            patch.object(experiment, "prepare_search", counted_prepare),
            patch.object(experiment, "aggregate_fixture_search", checked_aggregate),
        ):
            result = run_cell(manifest, key, "manifest-hash", "source-commit")

        self.assertEqual(prepared_modes, list(Mode))
        self.assertEqual(aggregate_modes, list(Mode))
        self.assertTrue(result.glv_control_passed)
        equal_work_prefixes = {
            payload["metrics"]["equal_primitive_work"]
            for payload in result.modes.values()
        }
        self.assertEqual(len(equal_work_prefixes), 1)
        self.assertEqual(result.glv_control_details["actual_targets_checked"], 4)
        self.assertGreater(result.glv_control_details["actual_relation_rows"], 0)

    def test_modes_retain_every_exact_registered_metric_and_primitive_projection(self):
        manifest = Manifest.from_mapping(mini_manifest(targets_per_cell=4))
        key = CellKey(6, 1, 2, 2, BudgetType.EQUAL_RAW)
        result = run_cell(manifest, key, "manifest-hash", "source-commit")
        expected = {metric for family in REGISTERED_METRIC_FAMILIES.values() for metric in family}

        for mode_payload in result.modes.values():
            registered = mode_payload["registered_metrics"]
            self.assertEqual(set(registered), expected)
            self.assertIsNone(
                registered["warm-wall-time-per-independent-relation"]
            )
            self.assertEqual(
                mode_payload["metric_status"]["warm-wall-time-per-independent-relation"],
                "censored-machine-dependent",
            )
            accepted = mode_payload["totals"]["accepted_relations"]
            vector = mode_payload["metrics"]["primitive_cost_per_accepted_relation"]
            projected = registered["primitive-cost-per-accepted-relation"]
            if accepted:
                self.assertAlmostEqual(projected, sum(vector.values()))
            else:
                self.assertIsNone(projected)

    def test_empty_search_records_zero_effects_and_one_authoritative_censor_ledger(self):
        import mcollapse.experiment as experiment

        manifest = Manifest.from_mapping(mini_manifest(targets_per_cell=2))
        key = CellKey(6, 1, 2, 2, BudgetType.EQUAL_RAW)

        def empty_search(prepared, target):
            return SearchResult((), CostVector(), 0)

        with patch.object(PreparedSearch, "search", empty_search):
            result = run_cell(manifest, key, "manifest-hash", "source-commit")

        observed_zero = {
            "independent-relations-per-unknown",
            "independent-relations-per-work",
            "decomposition-probability",
            "unique-normalized-vectors-per-target",
            "rank-increment-fraction",
        }
        ledger_total = 0
        for payload in result.modes.values():
            registered = payload["registered_metrics"]
            ledger = payload["censor_ledger"]
            ledger_total += len(ledger)
            for metric in observed_zero:
                self.assertEqual(registered[metric], 0.0)
                self.assertNotIn(metric, ledger)
            self.assertEqual(set(ledger), {
                "work-to-full-rank",
                "conditional-representation-multiplicity",
                "calibrated-cost-per-independent-relation",
                "primitive-cost-per-accepted-relation",
                "primitive-cost-per-independent-relation",
                "warm-wall-time-per-independent-relation",
            })
        self.assertEqual(
            sum(entry["category"] == "censored-metric" for entry in result.exception_log),
            ledger_total,
        )
        self.assertEqual(
            experiment._metric_ratio(
                "H2", "decomposition-probability", 0.0, 1.0
            ),
            0.0,
        )
        self.assertIsNone(
            experiment._metric_ratio(
                "H2", "decomposition-probability", 0.0, 0.0
            )
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest_path = write_manifest(root, mini_manifest(targets_per_cell=2))
            with patch.object(PreparedSearch, "search", empty_search):
                run = run_manifest(manifest_path, root / "output")
            self.assertEqual(run.summary["censored_metric_count"], ledger_total)

    def test_each_metric_gets_real_distinct_inference_and_holm_gating(self):
        import mcollapse.experiment as experiment

        manifest = Manifest.from_mapping(
            mini_manifest(fixture_seeds=list(range(1, 7)), bootstrap_iterations=50)
        )
        metric_names = REGISTERED_METRIC_FAMILIES["H1"]
        mcollapse_values = (
            (1.20, 1.20, 0.80, 0.80),
            (1.21, 1.10, 0.81, 0.90),
            (1.22, 1.05, 0.82, 1.10),
            (1.23, 0.95, 0.83, 1.20),
            (1.24, 0.90, 0.84, 1.30),
            (1.25, 0.85, 0.85, 1.40),
        )
        cells = []
        for seed, values in enumerate(mcollapse_values, 1):
            control_metrics = dict(zip(metric_names, (1.0, 1.0, 1.0, 1.0)))
            mcollapse_metrics = dict(zip(metric_names, values))
            for arity in (2, 3):
                cells.append(
                    SimpleNamespace(
                        status="complete",
                        key=CellKey(6, seed, 2, arity, BudgetType.EQUAL_RAW),
                        modes={
                            Mode.CLASSIC.value: {"registered_metrics": control_metrics},
                            Mode.MCOLLAPSE_WEIGHTED.value: {"registered_metrics": mcollapse_metrics},
                        },
                        glv_control_passed=True,
                    )
                )

        summary = experiment._comparison_summary(
            manifest,
            tuple(cells),
            "H1",
            BudgetType.EQUAL_RAW,
            Mode.MCOLLAPSE_WEIGHTED,
            Mode.CLASSIC,
            False,
        )
        metric_results = summary["metric_results"]
        self.assertEqual(set(metric_results), set(metric_names))
        self.assertGreater(len({item["raw_p_value"] for item in metric_results.values()}), 1)
        self.assertGreater(
            len({tuple(item["confidence_interval_95"]) for item in metric_results.values()}),
            1,
        )
        self.assertEqual(
            set(summary["holm_adjusted_p_values"]), set(metric_names)
        )
        self.assertEqual(
            summary["raw_p_values"][metric_names[0]], 1 / 64
        )

    def test_strongest_control_blocks_support_from_an_easy_control(self):
        import mcollapse.experiment as experiment

        comparisons = (
            {"control": "classic", "verdict": "supported in tested regime", "estimate": 1.30},
            {"control": "negation", "verdict": "supported in tested regime", "estimate": 1.20},
            {"control": "glv-expanded", "verdict": "inconclusive", "estimate": 1.01},
        )
        strongest = experiment._strongest_control_summary("H1", comparisons)
        self.assertEqual(strongest["strongest_control"], "glv-expanded")
        self.assertEqual(strongest["verdict"], "inconclusive")
        self.assertFalse(strongest["supported_against_all_controls"])

    def test_strongest_control_inference_is_not_replaced_by_easier_rejection(self):
        import mcollapse.experiment as experiment

        comparisons = (
            {
                "control": "classic", "verdict": "rejected in tested regime",
                "estimate": 1.20, "confidence_interval_95": [1.1, 1.3],
                "raw_p_values": {"m": 0.01}, "holm_adjusted_p_values": {"m": 0.04},
            },
            {
                "control": "glv-expanded", "verdict": "inconclusive",
                "estimate": 1.01, "confidence_interval_95": [0.9, 1.1],
                "raw_p_values": {"m": 0.4}, "holm_adjusted_p_values": {"m": 0.8},
            },
        )
        strongest = experiment._strongest_control_summary("H1", comparisons)
        self.assertEqual(strongest["strongest_control"], "glv-expanded")
        self.assertEqual(strongest["estimate"], 1.01)
        self.assertEqual(strongest["confidence_interval_95"], [0.9, 1.1])
        self.assertEqual(strongest["raw_p_values"], {"m": 0.4})
        self.assertEqual(strongest["verdict"], "inconclusive")

    def test_equal_work_uses_the_factor_base_candidate_limit_under_the_manifest_cap(self):
        import mcollapse.experiment as experiment

        manifest = Manifest.from_mapping(
            mini_manifest(budget_types=["equal-work"], candidate_limit=100)
        )
        key = CellKey(6, 1, 2, 2, BudgetType.EQUAL_WORK)
        original_prepare = experiment.prepare_search
        limits: list[int] = []

        def checked_prepare(fixture, base, arity, candidate_limit):
            limits.append(candidate_limit)
            return original_prepare(fixture, base, arity, candidate_limit)

        with patch.object(experiment, "prepare_search", checked_prepare):
            run_cell(manifest, key, "manifest-hash", "source-commit")
        self.assertEqual(limits, [2] * len(Mode))

    def test_fairness_uses_one_shared_primitive_work_prefix(self):
        manifest = Manifest.from_mapping(mini_manifest(targets_per_cell=4))
        key = CellKey(6, 1, 2, 2, BudgetType.EQUAL_RAW)
        result = run_cell(manifest, key, "manifest-hash", "source-commit")
        shared = {
            payload["metrics"]["equal_primitive_work"]
            for payload in result.modes.values()
        }
        self.assertEqual(len(shared), 1)
        for payload in result.modes.values():
            self.assertEqual(
                payload["registered_metrics"]["nullity-after-equal-candidates"],
                payload["metrics"]["nullity_after_equal_primitive_work"],
            )

    def test_partial_max_cells_is_recorded_and_cannot_support_a_hypothesis(self):
        payload = mini_manifest(fixture_seeds=[1, 2])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest_path = write_manifest(root, payload)
            result = run_manifest(manifest_path, root / "output", max_cells=1)
            self.assertTrue(result.partial)
            self.assertEqual(result.completed_cells, 1)
            self.assertEqual(result.total_cells, 2)
            summary = dict(result.summary)
            self.assertEqual(summary["max_cells"], 1)
            self.assertEqual(summary["censored_cells"], 1)
            self.assertFalse(summary["supports_any_hypothesis"])
            self.assertNotIn("supported in tested regime", (root / "output/report.md").read_text())

    def test_fixture_retry_preserves_slot_and_records_every_attempt(self):
        import mcollapse.experiment as experiment

        manifest = Manifest.from_mapping(mini_manifest(targets_per_cell=1))
        key = CellKey(6, 1, 2, 2, BudgetType.EQUAL_RAW)
        calls: list[int] = []
        ranking_seeds: list[int] = []
        original_build = experiment.build_comparison_bases

        def reject_once(bits: int, seed: int):
            calls.append(seed)
            if len(calls) == 1:
                raise RuntimeError("injected rejection")
            return generate_fixture(bits, seed)

        def capture_ranking_seed(fixture, budget_type, budget, seed):
            ranking_seeds.append(seed)
            return original_build(fixture, budget_type, budget, seed)

        with (
            patch.object(experiment, "generate_fixture", reject_once),
            patch.object(experiment, "build_comparison_bases", capture_ranking_seed),
        ):
            result = run_cell(manifest, key, "manifest-hash", "source-commit")
        self.assertEqual(result.status, "complete")
        self.assertEqual(len(result.fixture_attempts), 2)
        self.assertEqual(result.fixture_attempts[0]["reason"], "injected rejection")
        self.assertEqual(result.fixture_attempts[1]["reason"], "accepted")
        self.assertNotEqual(calls[0], calls[1])
        self.assertEqual(ranking_seeds, [calls[1]])
        self.assertEqual(result.fixture["requested_seed"], 1)
        self.assertEqual(result.fixture["accepted_seed"], calls[1])

    def test_accepted_fixture_slot_is_generated_once_for_all_dependent_cells(self):
        import mcollapse.experiment as experiment

        payload = mini_manifest(arities=[2, 3], targets_per_cell=1)
        calls: list[tuple[int, int]] = []
        original_generate = experiment.generate_fixture

        def counted_generate(bits: int, seed: int):
            calls.append((bits, seed))
            return original_generate(bits, seed)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest_path = write_manifest(root, payload)
            with (
                patch.object(experiment, "generate_fixture", counted_generate),
                patch.object(experiment, "_measure_warm_timings", return_value={}),
            ):
                result = run_manifest(manifest_path, root / "output")
            self.assertEqual(result.completed_cells, 2)
            self.assertEqual(calls, [(6, 1)])
            cells = [json.loads(path.read_text()) for path in (root / "output/cells").glob("*.json")]
            self.assertEqual(len({cell["fixture_resolution_id"] for cell in cells}), 1)

    def test_exhausted_fixture_retry_preserves_all_attempts_and_marks_shortfall(self):
        import mcollapse.experiment as experiment

        manifest = Manifest.from_mapping(mini_manifest(targets_per_cell=1))
        key = CellKey(6, 1, 2, 2, BudgetType.EQUAL_RAW)

        def always_fail(bits: int, seed: int):
            raise RuntimeError(f"rejected seed {seed}")

        with patch.object(experiment, "generate_fixture", always_fail):
            with self.assertRaises(experiment.FixtureAttemptsExhausted) as caught:
                run_cell(manifest, key, "manifest-hash", "source-commit")
        attempts = caught.exception.attempts
        self.assertEqual(len(attempts), 32)
        self.assertEqual(attempts[0]["seed"], 1)
        self.assertEqual(attempts[0]["reason"], "rejected seed 1")
        self.assertEqual(len({item["seed"] for item in attempts}), 32)
        with self.assertRaises(TypeError):
            attempts[0]["reason"] = "changed"

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest_path = write_manifest(root, mini_manifest(targets_per_cell=1))
            with patch.object(experiment, "generate_fixture", always_fail):
                result = run_manifest(manifest_path, root / "output")
            self.assertTrue(result.partial)
            self.assertEqual(result.completed_cells, 0)
            self.assertEqual(result.summary["failed_cells"], 1)
            self.assertEqual(result.summary["valid_fixture_shortfall"], 1)
            failure = json.loads(next((root / "output/failures").glob("*.json")).read_text())
            self.assertEqual(failure["category"], "fixture-attempts-exhausted")
            self.assertEqual(len(failure["fixture_attempts"]), 32)

    def test_one_exhausted_fixture_slot_skips_all_dependent_cells_once(self):
        import mcollapse.experiment as experiment

        payload = mini_manifest(
            factor_base_budgets=[2, 3], arities=[2, 3], targets_per_cell=1
        )
        calls: list[int] = []

        def always_fail(bits: int, seed: int):
            calls.append(seed)
            raise RuntimeError(f"rejected seed {seed}")

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest_path = write_manifest(root, payload)
            with patch.object(experiment, "generate_fixture", always_fail):
                result = run_manifest(manifest_path, root / "output")

            self.assertEqual(len(calls), 32)
            self.assertEqual(result.completed_cells, 0)
            self.assertTrue(result.partial)
            self.assertEqual(result.summary["valid_fixture_shortfall"], 1)
            self.assertEqual(result.summary["failed_fixture_slots"], 1)
            self.assertEqual(result.summary["skipped_cells"], 4)
            failures = tuple((root / "output/failures").glob("*.json"))
            self.assertEqual(len(failures), 1)
            failure = json.loads(failures[0].read_text())
            self.assertEqual(len(failure["fixture_attempts"]), 32)
            self.assertEqual(len(failure["dependent_cells"]), 4)
            self.assertEqual(
                {cell["status"] for cell in failure["dependent_cells"]},
                {"skipped-fixture-slot"},
            )

    def test_public_target_records_are_exactly_unique_and_capacity_is_checked(self):
        import mcollapse.experiment as experiment

        manifest = Manifest.from_mapping(mini_manifest(targets_per_cell=20))
        key = CellKey(6, 1, 2, 2, BudgetType.EQUAL_RAW)
        fixture = generate_fixture(6, 1)
        _, records, _ = experiment._make_public_targets_and_verifier(
            fixture, manifest, key
        )
        canonical = {
            (record["u"], record["v"], record["R"]) for record in records
        }
        self.assertEqual(len(canonical), 20)

        impossible = Manifest.from_mapping(mini_manifest(targets_per_cell=145))
        with self.assertRaisesRegex(ValueError, "distinct target capacity"):
            experiment._make_public_targets_and_verifier(tiny_fixture(), impossible, key)

    def test_canonical_mode_output_is_compact_and_timing_samples_at_most_three_targets(self):
        import mcollapse.experiment as experiment

        manifest = Manifest.from_mapping(mini_manifest(targets_per_cell=20))
        key = CellKey(6, 1, 2, 2, BudgetType.EQUAL_RAW)
        diagnostics: dict[str, object] = {}
        result = run_cell(
            manifest, key, "manifest-hash", "source-commit", diagnostics=diagnostics
        )
        for payload in result.modes.values():
            self.assertNotIn("target_records", payload)
            self.assertNotIn("equation_rows", payload)
            self.assertLessEqual(len(payload["target_audit"]), 3)
        for timing in diagnostics["modes"].values():
            self.assertEqual(timing["repeat_count"], 3)
            self.assertLessEqual(timing["target_count"], 3)

    def test_environment_metadata_and_git_discovery_are_cwd_independent(self):
        import mcollapse.experiment as experiment

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            previous = Path.cwd()
            try:
                os.chdir(root)
                self.assertEqual(experiment._git_root(), ROOT)
                manifest_path = write_manifest(root, mini_manifest(targets_per_cell=1))
                result = run_manifest(manifest_path, root / "output")
            finally:
                os.chdir(previous)
            self.assertEqual(result.summary["environment"], experiment._environment())
            cell = json.loads(next((root / "output/cells").glob("*.json")).read_text())
            self.assertEqual(cell["environment"], experiment._environment())

    def test_manifest_and_source_mutation_abort_during_run(self):
        import mcollapse.experiment as experiment

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest_path = write_manifest(root, mini_manifest(targets_per_cell=1))
            original_search = PreparedSearch.search
            changed = False

            def mutate_manifest(prepared, target):
                nonlocal changed
                result = original_search(prepared, target)
                if not changed:
                    changed = True
                    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
                    payload["name"] = "mutated"
                    manifest_path.write_text(json.dumps(payload), encoding="utf-8")
                return result

            with patch.object(PreparedSearch, "search", mutate_manifest):
                with self.assertRaisesRegex(RuntimeError, "manifest changed during run"):
                    run_manifest(manifest_path, root / "manifest-output")

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest_path = write_manifest(root, mini_manifest(targets_per_cell=1))
            with patch.object(experiment, "_source_identity", side_effect=("source-a", "source-b")):
                with self.assertRaisesRegex(RuntimeError, "source changed during run"):
                    run_manifest(manifest_path, root / "source-output")

    def test_interrupted_progress_resume_does_not_recompute_completed_targets(self):
        import mcollapse.experiment as experiment

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest_path = write_manifest(root, mini_manifest(targets_per_cell=3))
            output = root / "output"
            original_search = PreparedSearch.search
            calls = 0

            def interrupt(prepared, target):
                nonlocal calls
                calls += 1
                if calls == len(Mode) + 2:
                    raise RuntimeError("interrupt")
                return original_search(prepared, target)

            with (
                patch.object(experiment, "_CHECKPOINT_BATCH", 1),
                patch.object(experiment, "_measure_warm_timings", return_value={}),
                patch.object(PreparedSearch, "search", interrupt),
            ):
                with self.assertRaisesRegex(RuntimeError, "interrupt"):
                    run_manifest(manifest_path, output)

            resumed_calls = 0

            def count_resume(prepared, target):
                nonlocal resumed_calls
                resumed_calls += 1
                return original_search(prepared, target)

            with (
                patch.object(experiment, "_CHECKPOINT_BATCH", 1),
                patch.object(experiment, "_measure_warm_timings", return_value={}),
                patch.object(PreparedSearch, "search", count_resume),
            ):
                run_manifest(manifest_path, output, resume=True)
            self.assertEqual(resumed_calls, 10)

    def test_max_cells_marks_even_a_fully_scheduled_mini_manifest_as_pilot(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest_path = write_manifest(root, mini_manifest())
            result = run_manifest(manifest_path, root / "output", max_cells=1)
            self.assertTrue(result.partial)
            self.assertTrue(result.summary["pilot_censored"])
            self.assertFalse(result.summary["supports_any_hypothesis"])

    def test_outputs_are_atomic_and_report_contains_registered_sections(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "atomic.json"
            with patch("mcollapse.experiment.os.fsync") as fsync:
                atomic_json(target, {"value": 1})
            fsync.assert_called_once()
            self.assertEqual(json.loads(target.read_text()), {"value": 1})
            self.assertFalse(target.with_suffix(".json.tmp").exists())

            manifest_path = write_manifest(root, mini_manifest())
            output = root / "output"
            run_manifest(manifest_path, output)
            self.assertTrue((output / "manifest.json").is_file())
            self.assertTrue((output / "summary.json").is_file())
            report = (output / "report.md").read_text(encoding="utf-8")
            for required in (
                "H1", "H2", "H3", "equal-raw", "control", "raw p-value",
                "Holm", "95% CI", "orbit compression vs independent-relation yield",
                "normalized representation multiplicity vs rank-increment fraction",
                "calibrated cost vs independent-relation yield", "fixture count",
                "censored", "GLV/MCOLLAPSE", "H4-H6 were not tested",
                "strongest control", "machine-dependent diagnostics",
                "primitive scalar projection",
                "actual emitted relation rows",
            ):
                self.assertIn(required, report)

    def test_cli_runs_and_prints_hash_progress_and_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest_path = write_manifest(root, mini_manifest(targets_per_cell=1))
            output = root / "cli-output"
            with patch("builtins.print") as output_print:
                exit_code = main([
                    "--manifest", str(manifest_path), "--output", str(output),
                    "--max-cells", "1",
                ])
            self.assertEqual(exit_code, 0)
            line = output_print.call_args.args[0]
            self.assertIn("manifest=", line)
            self.assertIn("cells=1/1", line)
            self.assertIn(f"output={output}", line)

    def test_task7_modules_use_only_standard_library_and_mcollapse_imports(self):
        allowed_roots = {
            "__future__", "argparse", "collections", "dataclasses", "enum",
            "functools", "hashlib", "inspect", "itertools", "json", "math",
            "numbers", "os", "pathlib", "random", "statistics", "subprocess",
            "sys", "time", "types", "typing", "mcollapse",
        }
        for relative in ("mcollapse/experiment.py", "mcollapse/run_experiment.py"):
            source = (ROOT / relative).read_text(encoding="utf-8")
            tree = ast.parse(source)
            imports = {
                alias.name.split(".")[0]
                for node in ast.walk(tree)
                if isinstance(node, ast.Import)
                for alias in node.names
            }
            imports.update(
                node.module.split(".")[0]
                for node in ast.walk(tree)
                if isinstance(node, ast.ImportFrom) and node.module
            )
            self.assertLessEqual(imports, allowed_roots)


if __name__ == "__main__":
    unittest.main()
