"""Reproducible manifest execution for the MCOLLAPSE H1-H3 experiment."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, fields
from hashlib import sha256
from math import comb, isfinite
from numbers import Real
from pathlib import Path
from random import Random
from statistics import fmean
from types import MappingProxyType
from typing import Callable, Iterable, Mapping

from mcollapse.costs import CostVector
from mcollapse.curves import CurveFixture, Point, generate_fixture, validate_fixture
from mcollapse.factor_base import BudgetType, FactorBase, Mode, build_comparison_bases
from mcollapse.linear import (
    assert_glv_mcollapse_equivalent,
    canonical_row_space,
    eliminate_glv_coefficients,
    glv_constraint_rows,
    incremental_ranks,
)
from mcollapse.metrics import (
    H1_INDEPENDENT_RELATIONS_PER_UNKNOWN,
    H1_INDEPENDENT_RELATIONS_PER_WORK,
    H1_NULLITY_AFTER_EQUAL_CANDIDATES,
    H1_WORK_TO_FULL_RANK,
    H2_CONDITIONAL_REPRESENTATION_MULTIPLICITY,
    H2_DECOMPOSITION_PROBABILITY,
    H2_RANK_INCREMENT_FRACTION,
    H2_UNIQUE_NORMALIZED_VECTORS_PER_TARGET,
    H3_CALIBRATED_COST_PER_INDEPENDENT_RELATION,
    H3_PRIMITIVE_COST_PER_ACCEPTED_RELATION,
    H3_PRIMITIVE_COST_PER_INDEPENDENT_RELATION,
    H3_WARM_WALL_TIME_PER_INDEPENDENT_RELATION,
    REGISTERED_METRIC_FAMILIES,
    EffectSummary,
    FixtureObservation,
    aggregate_fixture_search,
    clustered_interval,
    correlation_report,
    evaluate_hypothesis,
)
from mcollapse.orbits import canonicalize
from mcollapse.relations import (
    PreparedSearch,
    Relation,
    SearchResult,
    prepare_search,
    verify_relation,
)


_MANIFEST_KEYS = frozenset(
    {
        "schema",
        "name",
        "subgroup_bits",
        "fixture_seeds",
        "factor_base_budgets",
        "arities",
        "budget_types",
        "targets_per_cell",
        "candidate_limit",
        "bootstrap_iterations",
        "bootstrap_seed",
        "calibration_weights",
    }
)
_COST_NAMES = tuple(field.name for field in fields(CostVector))
_CORRELATION_LABELS = (
    "orbit compression vs independent-relation yield",
    "normalized representation multiplicity vs rank-increment fraction",
    "calibrated cost vs independent-relation yield",
)
_CHECKPOINT_BATCH = 100
_FIXTURE_ATTEMPT_LIMIT = 32
_TIMING_REPEATS = 3
_AUDIT_SAMPLE_LIMIT = 3


class FixtureAttemptsExhausted(RuntimeError):
    """Raised with every immutable fixture attempt after the finite cap."""

    def __init__(self, attempts: Iterable[Mapping[str, object]]) -> None:
        self._attempts = tuple(
            MappingProxyType(dict(attempt)) for attempt in attempts
        )
        super().__init__(
            f"valid fixture attempt cap exhausted after {len(self._attempts)} attempts"
        )

    @property
    def attempts(self) -> tuple[Mapping[str, object], ...]:
        return self._attempts


@dataclass(frozen=True)
class FixtureResolution:
    subgroup_bits: int
    requested_seed: int
    fixture: CurveFixture
    attempts: tuple[Mapping[str, object], ...]

    def __post_init__(self) -> None:
        _strict_int(self.subgroup_bits, "subgroup_bits", minimum=2)
        _strict_int(self.requested_seed, "requested_seed", minimum=0)
        if not validate_fixture(self.fixture):
            raise ValueError("fixture resolution requires a valid fixture")
        if not isinstance(self.attempts, tuple) or not self.attempts:
            raise ValueError("fixture resolution attempts must be a nonempty tuple")
        frozen = tuple(MappingProxyType(dict(item)) for item in self.attempts)
        if frozen[-1].get("reason") != "accepted":
            raise ValueError("fixture resolution must end with an accepted attempt")
        if frozen[-1].get("seed") != self.fixture.seed:
            raise ValueError("accepted attempt seed must match fixture.seed")
        object.__setattr__(self, "attempts", frozen)

    @property
    def resolution_id(self) -> str:
        payload = {
            "subgroup_bits": self.subgroup_bits,
            "requested_seed": self.requested_seed,
            "accepted_seed": self.fixture.seed,
            "fixture": {
                "p": self.fixture.p,
                "b": self.fixture.b,
                "n": self.fixture.n,
                "generator": _point_json(self.fixture.generator),
                "beta": self.fixture.beta,
                "eigenvalue": self.fixture.eigenvalue,
                "seed": self.fixture.seed,
            },
            "attempts": _thaw(self.attempts),
        }
        canonical = json.dumps(
            payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode("ascii")
        return sha256(b"owic-fixture-resolution-v1\x00" + canonical).hexdigest()


def _freeze(value: object) -> object:
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    return value


def _thaw(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    return value


def _payload_hash(payload: Mapping[str, object]) -> str:
    """Hash a canonical JSON object while excluding its hash field."""

    material = dict(payload)
    material.pop("payload_hash", None)
    encoded = json.dumps(
        material, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")
    return sha256(encoded).hexdigest()


def _hashed_payload(payload: Mapping[str, object]) -> dict[str, object]:
    result = dict(payload)
    result["payload_hash"] = _payload_hash(result)
    return result


def _require_payload_hash(payload: Mapping[str, object], name: str) -> None:
    value = payload.get("payload_hash")
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
        or value != _payload_hash(payload)
    ):
        raise ValueError(f"{name} payload hash mismatch")


def _environment() -> dict[str, str]:
    return {
        "python_implementation": sys.implementation.name,
        "python_version": ".".join(str(value) for value in sys.version_info[:3]),
        "platform": sys.platform,
        "os_name": os.name,
        "byteorder": sys.byteorder,
    }


def _validate_environment(value: object) -> None:
    if not isinstance(value, Mapping) or set(value) != set(_environment()):
        raise ValueError("environment metadata has invalid fields")
    if not all(type(item) is str and item for item in value.values()):
        raise ValueError("environment metadata values must be nonempty strings")


def _strict_int(value: object, name: str, *, minimum: int | None = None) -> int:
    if type(value) is not int:
        raise TypeError(f"{name} must be an integer")
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return value


def _strict_string(value: object, name: str) -> str:
    if type(value) is not str:
        raise TypeError(f"{name} must be a string")
    if not value or value != value.strip():
        raise ValueError(f"{name} must be nonempty and trimmed")
    return value


def _unique_integer_tuple(
    value: object,
    name: str,
    *,
    minimum: int,
    maximum: int | None = None,
    allowed: frozenset[int] | None = None,
) -> tuple[int, ...]:
    if not isinstance(value, list) or not value:
        raise TypeError(f"{name} must be a nonempty JSON array")
    result = tuple(_strict_int(item, name, minimum=minimum) for item in value)
    if len(result) != len(set(result)):
        raise ValueError(f"{name} must not contain duplicate values")
    if maximum is not None and any(item > maximum for item in result):
        raise ValueError(f"{name} values must not exceed {maximum}")
    if allowed is not None and any(item not in allowed for item in result):
        raise ValueError(f"{name} contains an unregistered value")
    return result


@dataclass(frozen=True)
class Manifest:
    schema: int
    name: str
    subgroup_bits: tuple[int, ...]
    fixture_seeds: tuple[int, ...]
    factor_base_budgets: tuple[int, ...]
    arities: tuple[int, ...]
    budget_types: tuple[BudgetType, ...]
    targets_per_cell: int
    candidate_limit: int
    bootstrap_iterations: int
    bootstrap_seed: int
    calibration_weights: Mapping[str, float]

    def __post_init__(self) -> None:
        if self.schema != 1 or type(self.schema) is not int:
            raise ValueError("schema must be the integer 1")
        _strict_string(self.name, "name")
        if not self.subgroup_bits or any(
            type(item) is not int or not 2 <= item <= 16 for item in self.subgroup_bits
        ):
            raise ValueError("subgroup_bits must contain integers from 2 through 16")
        if len(self.subgroup_bits) != len(set(self.subgroup_bits)):
            raise ValueError("subgroup_bits must not contain duplicate values")
        for name in ("fixture_seeds", "factor_base_budgets", "arities"):
            values = getattr(self, name)
            if not values or any(type(item) is not int for item in values):
                raise ValueError(f"{name} must be a nonempty integer tuple")
            if len(values) != len(set(values)):
                raise ValueError(f"{name} must not contain duplicate values")
        if any(item < 0 for item in self.fixture_seeds):
            raise ValueError("fixture seeds must be nonnegative")
        if any(item <= 0 for item in self.factor_base_budgets):
            raise ValueError("factor-base budgets must be positive")
        if any(item not in (2, 3) for item in self.arities):
            raise ValueError("arities must contain only 2 and 3")
        if not self.budget_types or not all(
            isinstance(item, BudgetType) for item in self.budget_types
        ):
            raise ValueError("budget_types must be a nonempty BudgetType tuple")
        if len(self.budget_types) != len(set(self.budget_types)):
            raise ValueError("budget_types must not contain duplicate values")
        _strict_int(self.targets_per_cell, "targets_per_cell", minimum=1)
        _strict_int(self.candidate_limit, "candidate_limit", minimum=1)
        _strict_int(self.bootstrap_iterations, "bootstrap_iterations", minimum=1)
        _strict_int(self.bootstrap_seed, "bootstrap_seed")
        if not isinstance(self.calibration_weights, Mapping):
            raise TypeError("calibration_weights must be a mapping")
        if tuple(self.calibration_weights) != _COST_NAMES:
            raise ValueError("calibration_weights must contain every CostVector field in order")
        normalized: dict[str, float] = {}
        for name in _COST_NAMES:
            value = self.calibration_weights[name]
            if isinstance(value, bool) or not isinstance(value, Real):
                raise TypeError("calibration weights must be numeric")
            numeric = float(value)
            if not isfinite(numeric) or numeric < 0:
                raise ValueError("calibration weights must be finite and nonnegative")
            normalized[name] = numeric
        object.__setattr__(self, "calibration_weights", MappingProxyType(normalized))

    @classmethod
    def from_mapping(cls, payload: Mapping[str, object]) -> Manifest:
        if not isinstance(payload, Mapping):
            raise TypeError("manifest must be a JSON object")
        keys = set(payload)
        if keys != _MANIFEST_KEYS:
            missing = sorted(_MANIFEST_KEYS - keys)
            unknown = sorted(keys - _MANIFEST_KEYS)
            raise ValueError(f"manifest keys mismatch; missing={missing}, unknown={unknown}")
        budget_values = payload["budget_types"]
        if not isinstance(budget_values, list) or not budget_values:
            raise TypeError("budget_types must be a nonempty JSON array")
        if not all(type(item) is str for item in budget_values):
            raise TypeError("budget_types values must be strings")
        if len(budget_values) != len(set(budget_values)):
            raise ValueError("budget_types must not contain duplicate values")
        try:
            budget_types = tuple(BudgetType(item) for item in budget_values)
        except ValueError as error:
            raise ValueError("budget_types contains an unregistered value") from error
        weights = payload["calibration_weights"]
        if not isinstance(weights, Mapping):
            raise TypeError("calibration_weights must be a JSON object")
        if set(weights) != set(_COST_NAMES):
            raise ValueError("calibration_weights must contain exactly every CostVector field")
        ordered_weights = {name: weights[name] for name in _COST_NAMES}
        return cls(
            _strict_int(payload["schema"], "schema"),
            _strict_string(payload["name"], "name"),
            _unique_integer_tuple(
                payload["subgroup_bits"], "subgroup_bits", minimum=2, maximum=16
            ),
            _unique_integer_tuple(payload["fixture_seeds"], "fixture_seeds", minimum=0),
            _unique_integer_tuple(
                payload["factor_base_budgets"], "factor_base_budgets", minimum=1
            ),
            _unique_integer_tuple(
                payload["arities"], "arities", minimum=2, allowed=frozenset((2, 3))
            ),
            budget_types,
            _strict_int(payload["targets_per_cell"], "targets_per_cell", minimum=1),
            _strict_int(payload["candidate_limit"], "candidate_limit", minimum=1),
            _strict_int(
                payload["bootstrap_iterations"], "bootstrap_iterations", minimum=1
            ),
            _strict_int(payload["bootstrap_seed"], "bootstrap_seed"),
            ordered_weights,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "name": self.name,
            "subgroup_bits": list(self.subgroup_bits),
            "fixture_seeds": list(self.fixture_seeds),
            "factor_base_budgets": list(self.factor_base_budgets),
            "arities": list(self.arities),
            "budget_types": [item.value for item in self.budget_types],
            "targets_per_cell": self.targets_per_cell,
            "candidate_limit": self.candidate_limit,
            "bootstrap_iterations": self.bootstrap_iterations,
            "bootstrap_seed": self.bootstrap_seed,
            "calibration_weights": dict(self.calibration_weights),
        }

    @property
    def canonical_hash(self) -> str:
        encoded = json.dumps(
            self.to_dict(), sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode("ascii")
        return sha256(encoded).hexdigest()

    def cell_keys(self) -> tuple[CellKey, ...]:
        return tuple(
            CellKey(bits, seed, budget, arity, budget_type)
            for bits in self.subgroup_bits
            for seed in self.fixture_seeds
            for budget in self.factor_base_budgets
            for arity in self.arities
            for budget_type in self.budget_types
        )


@dataclass(frozen=True)
class CellKey:
    subgroup_bits: int
    fixture_seed: int
    factor_base_budget: int
    arity: int
    budget_type: BudgetType

    def __post_init__(self) -> None:
        bits = _strict_int(self.subgroup_bits, "subgroup_bits", minimum=2)
        if bits > 16:
            raise ValueError("subgroup_bits must not exceed 16")
        _strict_int(self.fixture_seed, "fixture_seed", minimum=0)
        _strict_int(self.factor_base_budget, "factor_base_budget", minimum=1)
        if _strict_int(self.arity, "arity") not in (2, 3):
            raise ValueError("arity must be 2 or 3")
        if not isinstance(self.budget_type, BudgetType):
            raise TypeError("budget_type must be a BudgetType")

    @property
    def cell_id(self) -> str:
        canonical = json.dumps(
            self.to_dict(), sort_keys=True, separators=(",", ":")
        ).encode("ascii")
        suffix = sha256(canonical).hexdigest()[:12]
        return (
            f"b{self.subgroup_bits:02d}-s{self.fixture_seed:08d}-"
            f"fb{self.factor_base_budget:03d}-a{self.arity}-"
            f"{self.budget_type.value}-{suffix}"
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "subgroup_bits": self.subgroup_bits,
            "fixture_seed": self.fixture_seed,
            "factor_base_budget": self.factor_base_budget,
            "arity": self.arity,
            "budget_type": self.budget_type.value,
        }

    @classmethod
    def from_mapping(cls, payload: Mapping[str, object]) -> CellKey:
        expected = {
            "subgroup_bits", "fixture_seed", "factor_base_budget", "arity", "budget_type"
        }
        if not isinstance(payload, Mapping) or set(payload) != expected:
            raise ValueError("cell key has invalid fields")
        try:
            budget_type = BudgetType(payload["budget_type"])
        except (TypeError, ValueError) as error:
            raise ValueError("cell key has an invalid budget type") from error
        return cls(
            payload["subgroup_bits"],  # type: ignore[arg-type]
            payload["fixture_seed"],  # type: ignore[arg-type]
            payload["factor_base_budget"],  # type: ignore[arg-type]
            payload["arity"],  # type: ignore[arg-type]
            budget_type,
        )


@dataclass(frozen=True)
class CellResult:
    key: CellKey
    manifest_hash: str
    source_commit: str
    source_identity: str
    fixture_resolution_id: str
    status: str
    fixture: Mapping[str, object]
    q: Point
    modes: Mapping[str, object]
    glv_control_passed: bool
    glv_control_details: Mapping[str, object]
    exception_log: tuple[Mapping[str, object], ...]
    short_orbits: tuple[Mapping[str, object], ...]
    fixture_attempts: tuple[Mapping[str, object], ...]
    environment: Mapping[str, object]

    def __post_init__(self) -> None:
        if not isinstance(self.key, CellKey):
            raise TypeError("key must be a CellKey")
        _strict_string(self.manifest_hash, "manifest_hash")
        _strict_string(self.source_commit, "source_commit")
        _strict_string(self.source_identity, "source_identity")
        _strict_string(self.fixture_resolution_id, "fixture_resolution_id")
        if self.status not in ("complete", "rejected"):
            raise ValueError("cell status must be complete or rejected")
        if type(self.glv_control_passed) is not bool:
            raise TypeError("glv_control_passed must be a boolean")
        if self.status == "complete" and not self.glv_control_passed:
            raise ValueError("a complete cell must pass the GLV control")
        for name in ("fixture", "modes", "glv_control_details", "environment"):
            if not isinstance(getattr(self, name), Mapping):
                raise TypeError(f"{name} must be a mapping")
        _validate_environment(self.environment)
        if not all(
            isinstance(value, tuple)
            for value in (self.exception_log, self.short_orbits, self.fixture_attempts)
        ):
            raise TypeError("cell logs must be tuples")
        object.__setattr__(self, "fixture", _freeze(self.fixture))
        object.__setattr__(self, "modes", _freeze(self.modes))
        object.__setattr__(self, "glv_control_details", _freeze(self.glv_control_details))
        object.__setattr__(self, "exception_log", _freeze(self.exception_log))
        object.__setattr__(self, "short_orbits", _freeze(self.short_orbits))
        object.__setattr__(self, "fixture_attempts", _freeze(self.fixture_attempts))
        object.__setattr__(self, "environment", _freeze(self.environment))

    def to_dict(self) -> dict[str, object]:
        return _hashed_payload({
            "cell_id": self.key.cell_id,
            "cell_key": self.key.to_dict(),
            "manifest_hash": self.manifest_hash,
            "source_commit": self.source_commit,
            "source_identity": self.source_identity,
            "fixture_resolution_id": self.fixture_resolution_id,
            "status": self.status,
            "fixture": _thaw(self.fixture),
            "q": _point_json(self.q),
            "modes": _thaw(self.modes),
            "glv_control_passed": self.glv_control_passed,
            "glv_control_details": _thaw(self.glv_control_details),
            "exception_log": _thaw(self.exception_log),
            "short_orbits": _thaw(self.short_orbits),
            "fixture_attempts": _thaw(self.fixture_attempts),
            "environment": _thaw(self.environment),
        })

    @classmethod
    def from_mapping(cls, payload: Mapping[str, object]) -> CellResult:
        required = {
            "cell_id", "cell_key", "manifest_hash", "source_commit", "source_identity",
            "fixture_resolution_id",
            "status", "fixture",
            "q", "modes", "glv_control_passed", "exception_log", "short_orbits",
            "glv_control_details", "fixture_attempts",
            "environment", "payload_hash",
        }
        if not isinstance(payload, Mapping) or set(payload) != required:
            raise ValueError("cell checkpoint schema mismatch")
        _require_payload_hash(payload, "cell checkpoint")
        key = CellKey.from_mapping(payload["cell_key"])  # type: ignore[arg-type]
        if payload["cell_id"] != key.cell_id:
            raise ValueError("cell checkpoint ID mismatch")
        logs = payload["exception_log"]
        short = payload["short_orbits"]
        attempts = payload["fixture_attempts"]
        if not all(isinstance(value, list) for value in (logs, short, attempts)):
            raise TypeError("cell checkpoint logs must be arrays")
        result = cls(
            key,
            payload["manifest_hash"],  # type: ignore[arg-type]
            payload["source_commit"],  # type: ignore[arg-type]
            payload["source_identity"],  # type: ignore[arg-type]
            payload["fixture_resolution_id"],  # type: ignore[arg-type]
            payload["status"],  # type: ignore[arg-type]
            payload["fixture"],  # type: ignore[arg-type]
            _point_from_json(payload["q"]),
            payload["modes"],  # type: ignore[arg-type]
            payload["glv_control_passed"],  # type: ignore[arg-type]
            payload["glv_control_details"],  # type: ignore[arg-type]
            tuple(logs),
            tuple(short),
            tuple(attempts),
            payload["environment"],  # type: ignore[arg-type]
        )
        _validate_cell_result_payload(result)
        return result


@dataclass(frozen=True)
class RunResult:
    manifest_hash: str
    source_commit: str
    completed_cells: int
    total_cells: int
    partial: bool
    summary: Mapping[str, object]

    def __post_init__(self) -> None:
        _strict_string(self.manifest_hash, "manifest_hash")
        _strict_string(self.source_commit, "source_commit")
        completed = _strict_int(self.completed_cells, "completed_cells", minimum=0)
        total = _strict_int(self.total_cells, "total_cells", minimum=0)
        if completed > total:
            raise ValueError("completed_cells cannot exceed total_cells")
        if type(self.partial) is not bool:
            raise TypeError("partial must be a boolean")
        if completed < total and not self.partial:
            raise ValueError("an incomplete run must be marked partial")
        if not isinstance(self.summary, Mapping):
            raise TypeError("summary must be a mapping")
        object.__setattr__(self, "summary", _freeze(self.summary))


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key: {key}")
        result[key] = value
    return result


def _read_json(path: Path) -> object:
    try:
        return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_reject_duplicate_keys)
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid JSON in {path}") from error


def load_manifest(path: Path | str) -> Manifest:
    manifest_path = Path(path)
    payload = _read_json(manifest_path)
    if not isinstance(payload, Mapping):
        raise TypeError("manifest root must be a JSON object")
    return Manifest.from_mapping(payload)


def atomic_json(path: Path | str, payload: Mapping[str, object]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(destination)


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _git_root() -> Path:
    anchor = Path(__file__).resolve().parents[1]
    try:
        value = subprocess.run(
            ("git", "rev-parse", "--show-toplevel"),
            cwd=anchor,
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as error:
        raise RuntimeError("source repository is unavailable") from error
    return Path(value)


def _source_commit() -> str:
    try:
        return subprocess.run(
            ("git", "rev-parse", "HEAD"),
            cwd=_git_root(),
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as error:
        raise RuntimeError("source commit is unavailable") from error


def _source_identity(ignore_paths: Iterable[Path] = ()) -> str:
    """Hash HEAD, tracked changes, and relevant untracked source files."""

    root = _git_root()
    ignored: list[Path] = []
    for path in ignore_paths:
        try:
            ignored.append(Path(path).resolve().relative_to(root.resolve()))
        except ValueError:
            continue
    try:
        head = subprocess.run(
            ("git", "rev-parse", "HEAD"),
            cwd=root,
            check=True,
            capture_output=True,
        ).stdout.strip()
        diff = subprocess.run(
            ("git", "diff", "--binary", "HEAD", "--", "."),
            cwd=root,
            check=True,
            capture_output=True,
        ).stdout
        untracked_output = subprocess.run(
            ("git", "ls-files", "--others", "--exclude-standard"),
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
        ).stdout
    except (OSError, subprocess.CalledProcessError) as error:
        raise RuntimeError("source identity is unavailable") from error
    digest = sha256(b"owic-source-identity-v1\x00" + head + b"\x00" + diff)
    relevant_suffixes = {".py", ".json", ".md", ".toml", ".yaml", ".yml"}
    for name in sorted(filter(None, untracked_output.splitlines())):
        relative = Path(name)
        if relative.suffix.lower() not in relevant_suffixes:
            continue
        if any(relative == item or item in relative.parents for item in ignored):
            continue
        path = root / relative
        digest.update(name.replace("\\", "/").encode("utf-8"))
        digest.update(b"\x00")
        digest.update(path.read_bytes())
        digest.update(b"\x00")
    return f"{head.decode('ascii')}+sha256:{digest.hexdigest()}"


def _assert_run_identity(
    manifest_path: Path,
    expected_manifest_hash: str,
    expected_source_identity: str,
    output_path: Path,
) -> None:
    if load_manifest(manifest_path).canonical_hash != expected_manifest_hash:
        raise RuntimeError("manifest changed during run")
    if _source_identity((output_path,)) != expected_source_identity:
        raise RuntimeError("source changed during run")


def _derived_fixture_seed(
    subgroup_bits: int, requested_seed: int, attempt: int
) -> int:
    if attempt == 0:
        return requested_seed
    encoded = json.dumps(
        {
            "bits": subgroup_bits,
            "requested_seed": requested_seed,
            "attempt": attempt,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")
    return int.from_bytes(
        sha256(b"owic-fixture-retry-v1\x00" + encoded).digest()[:8], "big"
    )


def _resolve_fixture_slot(
    subgroup_bits: int, requested_seed: int
) -> FixtureResolution:
    attempts: list[Mapping[str, object]] = []
    for attempt in range(_FIXTURE_ATTEMPT_LIMIT):
        seed = _derived_fixture_seed(subgroup_bits, requested_seed, attempt)
        try:
            fixture = generate_fixture(subgroup_bits, seed)
        except (ArithmeticError, RuntimeError, TypeError, ValueError) as error:
            attempts.append({"attempt": attempt, "seed": seed, "reason": str(error)})
            continue
        if not validate_fixture(fixture):
            attempts.append(
                {"attempt": attempt, "seed": seed, "reason": "fixture validation failed"}
            )
            continue
        attempts.append({"attempt": attempt, "seed": seed, "reason": "accepted"})
        return FixtureResolution(
            subgroup_bits, requested_seed, fixture, tuple(attempts)
        )
    raise FixtureAttemptsExhausted(attempts)


def _point_json(point: Point) -> list[int] | None:
    return None if point is None else [point[0], point[1]]


def _point_from_json(value: object) -> Point:
    if value is None:
        return None
    if (
        not isinstance(value, list)
        or len(value) != 2
        or any(type(item) is not int for item in value)
    ):
        raise ValueError("serialized point must be null or an integer pair")
    return value[0], value[1]


def _fixture_dict(
    fixture: CurveFixture, requested_seed: int | None = None
) -> dict[str, object]:
    result = {
        "p": fixture.p,
        "b": fixture.b,
        "n": fixture.n,
        "generator": _point_json(fixture.generator),
        "beta": fixture.beta,
        "eigenvalue": fixture.eigenvalue,
        "seed": fixture.seed,
    }
    if requested_seed is not None:
        result.update(
            {
                "requested_seed": requested_seed,
                "accepted_seed": fixture.seed,
            }
        )
    return result


def _cost_dict(cost: CostVector) -> dict[str, int]:
    return {name: getattr(cost, name) for name in _COST_NAMES}


def _cost_from(value: object) -> CostVector:
    if not isinstance(value, Mapping) or set(value) != set(_COST_NAMES):
        raise ValueError("serialized cost vector has invalid fields")
    return CostVector(*(value[name] for name in _COST_NAMES))  # type: ignore[arg-type]


def _target_seed(manifest: Manifest, key: CellKey) -> int:
    material = json.dumps(
        {"bootstrap_seed": manifest.bootstrap_seed, **key.to_dict()},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")
    return int.from_bytes(sha256(b"owic-targets-v1\x00" + material).digest()[:16], "big")


def _make_public_targets_and_verifier(
    fixture: CurveFixture, manifest: Manifest, key: CellKey
) -> tuple[tuple[int, int], tuple[Mapping[str, object], ...], Callable[..., bool]]:
    """Create public data while retaining all scalar oracle state in one closure."""

    rng = Random(_target_seed(manifest, key))
    hidden_scalar = rng.randrange(1, fixture.n)
    q = fixture.curve.mul(hidden_scalar, fixture.generator)
    if q is None:
        raise ArithmeticError("generated public Q is the identity")
    capacity = (fixture.n - 1) ** 2
    if manifest.targets_per_cell > capacity:
        raise ValueError(
            f"distinct target capacity {capacity} is below requested count "
            f"{manifest.targets_per_cell}"
        )
    targets: list[Mapping[str, object]] = []
    seen: set[tuple[int, int, tuple[int, int]]] = set()
    attempt_limit = max(1_000, manifest.targets_per_cell * 100)
    attempts = 0
    while len(targets) < manifest.targets_per_cell:
        attempts += 1
        if attempts > attempt_limit:
            raise RuntimeError("distinct public target attempt cap exhausted")
        u = rng.randrange(fixture.n)
        v = rng.randrange(1, fixture.n)
        target_scalar = (u + v * hidden_scalar) % fixture.n
        if target_scalar == 0:
            u = (u + 1) % fixture.n
            target_scalar = (u + v * hidden_scalar) % fixture.n
        target = fixture.curve.mul(target_scalar, fixture.generator)
        if target is None:
            raise ArithmeticError("public target generation reached the identity")
        canonical = (u, v, target)
        if canonical in seen:
            continue
        seen.add(canonical)
        targets.append(
            MappingProxyType({"index": len(targets), "u": u, "v": v, "R": target})
        )

    discrete_logs: dict[tuple[int, int], int] = {}
    point: Point = None
    for scalar in range(fixture.n):
        if point is not None:
            discrete_logs[point] = scalar
        point = fixture.curve.add(point, fixture.generator)

    def verify_equation(
        basis_points: tuple[tuple[int, int], ...],
        coefficients: tuple[int, ...],
        u: int,
        v: int,
    ) -> bool:
        if len(basis_points) != len(coefficients):
            return False
        try:
            left = sum(
                coefficient * discrete_logs[basis_point]
                for coefficient, basis_point in zip(
                    coefficients, basis_points, strict=True
                )
            )
        except KeyError:
            return False
        return (left - v * hidden_scalar - u) % fixture.n == 0

    return q, tuple(targets), verify_equation


def _aligned_control_rows(
    expanded: PreparedSearch, compact: PreparedSearch
) -> tuple[tuple[tuple[int, ...], ...], tuple[tuple[int, ...], ...], int]:
    if expanded.base.mode is not Mode.GLV_EXPANDED:
        raise ValueError("expanded control must use glv-expanded mode")
    if compact.base.mode is not Mode.MCOLLAPSE_UNWEIGHTED:
        raise ValueError("compact control must use mcollapse-unweighted mode")
    if expanded.base.representatives != compact.base.representatives:
        raise ArithmeticError("GLV/MCOLLAPSE representative alignment mismatch")
    if expanded.base.raw_points != compact.base.raw_points:
        raise ArithmeticError("GLV/MCOLLAPSE authorized observations mismatch")
    if not expanded.catalog.terms or not compact.catalog.terms:
        raise ArithmeticError("GLV/MCOLLAPSE control catalogs must be nonempty")

    expanded_by_point = {term.point: term for term in expanded.catalog.terms}
    compact_by_point = {term.point: term for term in compact.catalog.terms}
    if len(expanded_by_point) != len(expanded.catalog.terms):
        raise ArithmeticError("GLV control observations are not representative-aligned")
    if len(compact_by_point) != len(compact.catalog.terms):
        raise ArithmeticError("MCOLLAPSE control observations are not representative-aligned")
    if set(expanded_by_point) != set(compact_by_point):
        raise ArithmeticError("GLV/MCOLLAPSE control observations are not aligned")
    if tuple(term.point for term in expanded.catalog.terms) != tuple(
        term.point for term in compact.catalog.terms
    ):
        raise ArithmeticError("GLV/MCOLLAPSE canonical candidate ordering mismatch")

    expanded_rows: list[tuple[int, ...]] = []
    compact_rows: list[tuple[int, ...]] = []
    for point in sorted(expanded_by_point):
        expanded_term = expanded_by_point[point]
        compact_term = compact_by_point[point]
        expanded_row = [0] * expanded.column_count
        expanded_row[expanded_term.column] = expanded_term.coefficient % expanded.fixture.n
        compact_row = [0] * compact.column_count
        compact_row[compact_term.column] = compact_term.coefficient % compact.fixture.n
        expanded_rows.append(tuple(expanded_row))
        compact_rows.append(tuple(compact_row))
    return tuple(expanded_rows), tuple(compact_rows), len(compact.base.representatives)


def _actual_relation_control_rows(
    expanded: PreparedSearch,
    expanded_result: SearchResult,
    compact: PreparedSearch,
    compact_result: SearchResult,
    target: Mapping[str, object],
) -> tuple[tuple[tuple[int, ...], ...], tuple[tuple[int, ...], ...], int]:
    """Return only emitted relation rows with the same public RHS metadata."""

    if expanded.base.representatives != compact.base.representatives:
        raise ArithmeticError("GLV/MCOLLAPSE representative alignment mismatch")
    if not all(isinstance(item, Relation) for item in expanded_result.relations):
        raise TypeError("expanded control rows must come from emitted Relation objects")
    if not all(isinstance(item, Relation) for item in compact_result.relations):
        raise TypeError("compact control rows must come from emitted Relation objects")
    u = target["u"]
    v = target["v"]
    if type(u) is not int or type(v) is not int:
        raise ArithmeticError("public target metadata changed type")
    suffix = ((-v) % expanded.fixture.n, u % expanded.fixture.n)
    return (
        tuple(
            relation.normalized_row + suffix
            for relation in expanded_result.relations
        ),
        tuple(
            relation.normalized_row + suffix
            for relation in compact_result.relations
        ),
        len(compact.base.representatives),
    )


def _relation_metric_row(
    prepared: PreparedSearch, relation: Relation, v: int
) -> tuple[int, ...]:
    coefficients = relation.normalized_row
    if prepared.base.mode is Mode.GLV_EXPANDED:
        coefficients = tuple(
            eliminate_glv_coefficients(
                (coefficients,), prepared.fixture.eigenvalue, prepared.fixture.n
            )[0]
        )
    return coefficients + ((-v) % prepared.fixture.n,)


def _divide_cost(cost: CostVector, denominator: int) -> dict[str, float] | None:
    if denominator == 0:
        return None
    return {name: getattr(cost, name) / denominator for name in _COST_NAMES}


def _primitive_work(cost: CostVector) -> int:
    return sum(getattr(cost, name) for name in _COST_NAMES)


def _new_mode_state(prepared: PreparedSearch) -> dict[str, object]:
    return {
        "processed_targets": 0,
        "total_cost": _cost_dict(prepared.preparation_cost),
        "total_candidates": prepared.preparation_candidates,
        "accepted_relations": 0,
        "decomposed_targets": 0,
        "rank_basis": [],
        "rank_increments": 0,
        "full_rank_work": None,
        "rank_work_thresholds": [],
        "target_audit": [],
        "equation_audit": [],
    }


def _update_mode_state(
    state: dict[str, object],
    prepared: PreparedSearch,
    target: Mapping[str, object],
    result: SearchResult,
    calibration_weights: Mapping[str, float],
) -> None:
    index = state["processed_targets"]
    if type(index) is not int or index != target["index"]:
        raise ValueError("mode state target index is not monotone")
    total_cost = _cost_from(state["total_cost"]) + result.search_cost
    total_candidates = state["total_candidates"]
    accepted = state["accepted_relations"]
    decomposed = state["decomposed_targets"]
    increments = state["rank_increments"]
    if not all(type(value) is int for value in (
        total_candidates, accepted, decomposed, increments
    )):
        raise TypeError("mode state counters must be integers")
    total_candidates += result.search_candidates
    accepted += len(result.relations)
    decomposed += bool(result.relations)
    basis_value = state["rank_basis"]
    if not isinstance(basis_value, list):
        raise TypeError("mode state rank basis must be an array")
    basis = tuple(tuple(row) for row in basis_value)
    v = target["v"]
    u = target["u"]
    if type(v) is not int or type(u) is not int:
        raise ArithmeticError("public target metadata changed type")
    equation_audit = state["equation_audit"]
    if not isinstance(equation_audit, list):
        raise TypeError("mode state equation audit must be an array")
    thresholds = state["rank_work_thresholds"]
    if not isinstance(thresholds, list):
        raise TypeError("mode state rank work thresholds must be an array")
    cumulative_primitive_work = _primitive_work(total_cost)
    for relation in result.relations:
        row = _relation_metric_row(prepared, relation, v)
        updated = canonical_row_space(basis + (row,), prepared.fixture.n)
        if len(updated) > len(basis):
            increments += 1
            thresholds.append(cumulative_primitive_work)
        basis = updated
        if len(equation_audit) < _AUDIT_SAMPLE_LIMIT:
            equation_audit.append(
                {
                    "target_index": index,
                    "coefficients": list(relation.normalized_row),
                    "q_coefficient": (-v) % prepared.fixture.n,
                    "rhs": u,
                }
            )
    unknowns = prepared.base.effective_columns + 1
    full_rank_work = state["full_rank_work"]
    if full_rank_work is None and len(basis) >= unknowns:
        full_rank_work = total_cost.calibrated(calibration_weights)
    target_audit = state["target_audit"]
    if not isinstance(target_audit, list):
        raise TypeError("mode state audits must be arrays")
    if len(target_audit) < _AUDIT_SAMPLE_LIMIT:
        target_audit.append(
            {
                "index": index,
                "u": u,
                "v": v,
                "R": _point_json(target["R"]),  # type: ignore[arg-type]
                "search_candidates": result.search_candidates,
                "accepted_relations": len(result.relations),
                "search_cost": _cost_dict(result.search_cost),
            }
        )
    state.update(
        {
            "processed_targets": index + 1,
            "total_cost": _cost_dict(total_cost),
            "total_candidates": total_candidates,
            "accepted_relations": accepted,
            "decomposed_targets": decomposed,
            "rank_basis": [list(row) for row in basis],
            "rank_increments": increments,
            "full_rank_work": full_rank_work,
        }
    )


def _validate_mode_state(
    value: object,
    prepared: PreparedSearch,
    targets: tuple[Mapping[str, object], ...],
    next_target_index: int,
) -> dict[str, object]:
    expected = {
        "processed_targets", "total_cost", "total_candidates",
        "accepted_relations", "decomposed_targets", "rank_basis",
        "rank_increments", "full_rank_work", "rank_work_thresholds",
        "target_audit", "equation_audit",
    }
    if not isinstance(value, Mapping) or set(value) != expected:
        raise ValueError("progress mode aggregate has invalid fields")
    state = _thaw(value)
    if not isinstance(state, dict):
        raise TypeError("progress mode aggregate must be an object")
    if state["processed_targets"] != next_target_index:
        raise ValueError("progress mode target index is not monotone")
    for name in (
        "processed_targets", "total_candidates", "accepted_relations",
        "decomposed_targets", "rank_increments",
    ):
        _strict_int(state[name], f"progress {name}", minimum=0)
    total_cost = _cost_from(state["total_cost"])
    if state["total_candidates"] < prepared.preparation_candidates:
        raise ValueError("progress total candidates omit preparation")
    if state["decomposed_targets"] > next_target_index:
        raise ValueError("progress decomposed target count exceeds processed targets")
    basis = state["rank_basis"]
    if not isinstance(basis, list) or any(
        not isinstance(row, list)
        or len(row) != prepared.base.effective_columns + 1
        or any(type(item) is not int for item in row)
        for row in basis
    ):
        raise ValueError("progress rank basis is invalid")
    canonical = canonical_row_space(basis, prepared.fixture.n)
    if [list(row) for row in canonical] != basis:
        raise ValueError("progress rank basis is not canonical")
    if state["rank_increments"] != len(basis) or state["accepted_relations"] < len(basis):
        raise ValueError("progress rank counters disagree with the canonical basis")
    thresholds = state["rank_work_thresholds"]
    if not isinstance(thresholds, list) or len(thresholds) != len(basis):
        raise ValueError("progress rank work thresholds are invalid")
    previous_work = -1
    for threshold in thresholds:
        work = _strict_int(threshold, "primitive work threshold", minimum=0)
        if work < previous_work:
            raise ValueError("progress primitive work thresholds are not monotone")
        previous_work = work
    if thresholds and thresholds[-1] > _primitive_work(total_cost):
        raise ValueError("progress primitive work threshold exceeds total work")
    audits = state["target_audit"]
    if not isinstance(audits, list) or len(audits) != min(
        next_target_index, _AUDIT_SAMPLE_LIMIT
    ):
        raise ValueError("progress target audit is invalid")
    for index, audit in enumerate(audits):
        if not isinstance(audit, Mapping) or set(audit) != {
            "index", "u", "v", "R", "search_candidates",
            "accepted_relations", "search_cost",
        } or audit.get("index") != index:
            raise ValueError("progress target audit index is not monotone")
        target = targets[index]
        if (
            audit.get("u") != target["u"]
            or audit.get("v") != target["v"]
            or _point_from_json(audit.get("R")) != target["R"]
        ):
            raise ValueError("progress target audit does not match public data")
        _strict_int(audit["search_candidates"], "audit search candidates", minimum=0)
        _strict_int(audit["accepted_relations"], "audit accepted relations", minimum=0)
        _cost_from(audit["search_cost"])
    equations = state["equation_audit"]
    if not isinstance(equations, list) or len(equations) != min(
        state["accepted_relations"], _AUDIT_SAMPLE_LIMIT
    ):
        raise ValueError("progress equation audit is invalid")
    for equation in equations:
        if not isinstance(equation, Mapping) or set(equation) != {
            "target_index", "coefficients", "q_coefficient", "rhs"
        }:
            raise ValueError("progress equation audit has invalid fields")
        target_index = _strict_int(
            equation["target_index"], "equation target index", minimum=0
        )
        if target_index >= next_target_index:
            raise ValueError("progress equation target index exceeds processed targets")
        coefficients = equation["coefficients"]
        if not isinstance(coefficients, list) or len(coefficients) != prepared.column_count or any(
            type(item) is not int for item in coefficients
        ):
            raise ValueError("progress equation coefficients are invalid")
        _strict_int(equation["q_coefficient"], "equation Q coefficient", minimum=0)
        _strict_int(equation["rhs"], "equation RHS", minimum=0)
    full_rank_work = state["full_rank_work"]
    if full_rank_work is not None and (
        isinstance(full_rank_work, bool)
        or not isinstance(full_rank_work, Real)
        or not isfinite(full_rank_work)
        or full_rank_work < 0
    ):
        raise ValueError("progress full-rank work is invalid")
    return state


def _state_mode_payload(
    prepared: PreparedSearch,
    state: Mapping[str, object],
    calibration_weights: Mapping[str, float],
) -> dict[str, object]:
    target_count = state["processed_targets"]
    accepted = state["accepted_relations"]
    decomposed = state["decomposed_targets"]
    increments = state["rank_increments"]
    basis = state["rank_basis"]
    if not all(type(value) is int for value in (
        target_count, accepted, decomposed, increments
    )) or not isinstance(basis, list):
        raise TypeError("mode aggregate counters are invalid")
    total_cost = _cost_from(state["total_cost"])
    calibrated = total_cost.calibrated(calibration_weights)
    independent = len(basis)
    unknowns = prepared.base.effective_columns + 1
    primitive_total = _primitive_work(total_cost)
    independent_per_work = None if calibrated == 0 else independent / calibrated
    conditional = None if decomposed == 0 else accepted / decomposed
    primitive_per_accepted = None if accepted == 0 else primitive_total / accepted
    primitive_per_independent = None if independent == 0 else primitive_total / independent
    calibrated_per_independent = None if independent == 0 else calibrated / independent
    if target_count <= 0:
        raise ValueError("mode aggregate targets must be positive")
    metrics = {
        "independent_relations": independent,
        "effective_unknowns": unknowns,
        "independent_relations_per_unknown": independent / unknowns,
        "independent_relations_per_work": independent_per_work,
        "work_to_full_rank": state["full_rank_work"],
        "nullity_after_equal_candidates": max(0, unknowns - independent),
        "decomposition_probability": decomposed / target_count,
        "unique_rows_per_target": accepted / target_count,
        "rank_increment_fraction": 0.0 if accepted == 0 else increments / accepted,
        "conditional_representation_multiplicity": conditional,
        "primitive_cost_per_accepted_relation": _divide_cost(total_cost, accepted),
        "primitive_cost_per_independent_relation": _divide_cost(total_cost, independent),
        "calibrated_cost_per_accepted_relation": None if accepted == 0 else calibrated / accepted,
        "calibrated_cost_per_independent_relation": calibrated_per_independent,
        "primitive_scalar_projection": "sum of all CostVector primitive counters",
    }
    registered = {
        H1_INDEPENDENT_RELATIONS_PER_UNKNOWN: independent / unknowns,
        H1_INDEPENDENT_RELATIONS_PER_WORK: independent_per_work,
        H1_WORK_TO_FULL_RANK: state["full_rank_work"],
        H1_NULLITY_AFTER_EQUAL_CANDIDATES: max(0, unknowns - independent),
        H2_DECOMPOSITION_PROBABILITY: decomposed / target_count,
        H2_UNIQUE_NORMALIZED_VECTORS_PER_TARGET: accepted / target_count,
        H2_RANK_INCREMENT_FRACTION: 0.0 if accepted == 0 else increments / accepted,
        H2_CONDITIONAL_REPRESENTATION_MULTIPLICITY: conditional,
        H3_CALIBRATED_COST_PER_INDEPENDENT_RELATION: calibrated_per_independent,
        H3_PRIMITIVE_COST_PER_ACCEPTED_RELATION: primitive_per_accepted,
        H3_PRIMITIVE_COST_PER_INDEPENDENT_RELATION: primitive_per_independent,
        H3_WARM_WALL_TIME_PER_INDEPENDENT_RELATION: None,
    }
    reasons = {
        H1_INDEPENDENT_RELATIONS_PER_WORK: "zero calibrated work denominator",
        H1_WORK_TO_FULL_RANK: "full rank not reached",
        H2_CONDITIONAL_REPRESENTATION_MULTIPLICITY: "no decomposed targets",
        H3_CALIBRATED_COST_PER_INDEPENDENT_RELATION: "no independent relations",
        H3_PRIMITIVE_COST_PER_ACCEPTED_RELATION: "no accepted relations",
        H3_PRIMITIVE_COST_PER_INDEPENDENT_RELATION: "no independent relations",
        H3_WARM_WALL_TIME_PER_INDEPENDENT_RELATION: "machine-dependent diagnostic",
    }
    ledger = {
        name: reasons.get(name, "undefined registered metric")
        for name, value in registered.items() if value is None
    }
    status = {
        name: ("censored" if name in ledger else "observed") for name in registered
    }
    status[H3_WARM_WALL_TIME_PER_INDEPENDENT_RELATION] = "censored-machine-dependent"
    return {
        "mode": prepared.base.mode.value,
        "base": {
            "representatives": [_point_json(point) for point in prepared.base.representatives],
            "raw_points": [_point_json(point) for point in prepared.base.raw_points],
            "raw_point_count": len(prepared.base.raw_points),
            "effective_columns": prepared.base.effective_columns,
            "basis_points": [_point_json(point) for point in prepared.basis_points],
        },
        "preparation": {
            "cost": _cost_dict(prepared.preparation_cost),
            "candidates": prepared.preparation_candidates,
            "charged_count": 1,
        },
        "totals": {
            "cost": _cost_dict(total_cost),
            "calibrated_cost": calibrated,
            "primitive_work": primitive_total,
            "candidates": state["total_candidates"],
            "targets": target_count,
            "accepted_relations": accepted,
        },
        "metrics": metrics,
        "registered_metrics": registered,
        "metric_status": status,
        "censor_ledger": ledger,
        "target_audit": _thaw(state["target_audit"]),
        "equation_audit": _thaw(state["equation_audit"]),
    }


def _mode_payload(
    prepared: PreparedSearch,
    targets: tuple[Mapping[str, object], ...],
    results: tuple[SearchResult, ...],
    calibration_weights: Mapping[str, float],
) -> dict[str, object]:
    aggregate = aggregate_fixture_search(prepared, results, calibration_weights)
    metric_rows: list[tuple[int, ...]] = []
    equation_rows: list[dict[str, object]] = []
    decomposed = 0
    for target, result in zip(targets, results, strict=True):
        if result.relations:
            decomposed += 1
        for relation in result.relations:
            v = target["v"]
            u = target["u"]
            if type(v) is not int or type(u) is not int:
                raise ArithmeticError("public target metadata changed type")
            metric_rows.append(_relation_metric_row(prepared, relation, v))
            equation_rows.append(
                {
                    "coefficients": list(relation.normalized_row),
                    "q_coefficient": (-v) % prepared.fixture.n,
                    "rhs": u,
                    "u": u,
                    "v": v,
                    "R": _point_json(target["R"]),  # type: ignore[arg-type]
                }
            )
    ranks = incremental_ranks(metric_rows, prepared.fixture.n)
    independent = ranks[-1] if ranks else 0
    rank_increments = sum(
        rank > (ranks[index - 1] if index else 0) for index, rank in enumerate(ranks)
    )
    unknowns = prepared.base.effective_columns + 1
    full_rank_work: float | None = None
    running_cost = prepared.preparation_cost.calibrated(calibration_weights)
    relation_index = 0
    for result in results:
        running_cost += result.search_cost.calibrated(calibration_weights)
        for _ in result.relations:
            if ranks[relation_index] >= unknowns and full_rank_work is None:
                full_rank_work = running_cost
            relation_index += 1

    accepted = aggregate.accepted_relations
    calibrated = aggregate.calibrated_cost
    if calibrated is None:
        raise ArithmeticError("registered calibration was not applied")
    raw_count = len(prepared.base.raw_points)
    primitive_total = sum(getattr(aggregate.total_cost, name) for name in _COST_NAMES)
    independent_per_work = None if calibrated == 0 else independent / calibrated
    rank_increment_fraction = 0.0 if accepted == 0 else rank_increments / accepted
    conditional_multiplicity = None if decomposed == 0 else accepted / decomposed
    primitive_per_accepted = None if accepted == 0 else primitive_total / accepted
    primitive_per_independent = (
        None if independent == 0 else primitive_total / independent
    )
    calibrated_per_independent = (
        None if independent == 0 else calibrated / independent
    )
    metrics = {
        "independent_relations": independent,
        "effective_unknowns": unknowns,
        "independent_relations_per_unknown": independent / unknowns,
        "independent_relations_per_work": independent_per_work,
        "work_to_full_rank": full_rank_work,
        "nullity_after_equal_candidates": max(0, unknowns - independent),
        "decomposition_probability": decomposed / len(targets),
        "unique_rows_per_target": accepted / len(targets),
        "rank_increment_fraction": rank_increment_fraction,
        "conditional_representation_multiplicity": conditional_multiplicity,
        "primitive_cost_per_accepted_relation": _divide_cost(aggregate.total_cost, accepted),
        "primitive_cost_per_independent_relation": _divide_cost(
            aggregate.total_cost, independent
        ),
        "calibrated_cost_per_accepted_relation": (
            None if accepted == 0 else calibrated / accepted
        ),
        "calibrated_cost_per_independent_relation": calibrated_per_independent,
        "primitive_scalar_projection": "sum of all CostVector primitive counters",
    }
    registered_metrics = {
        H1_INDEPENDENT_RELATIONS_PER_UNKNOWN: independent / unknowns,
        H1_INDEPENDENT_RELATIONS_PER_WORK: independent_per_work,
        H1_WORK_TO_FULL_RANK: full_rank_work,
        H1_NULLITY_AFTER_EQUAL_CANDIDATES: max(0, unknowns - independent),
        H2_DECOMPOSITION_PROBABILITY: decomposed / len(targets),
        H2_UNIQUE_NORMALIZED_VECTORS_PER_TARGET: accepted / len(targets),
        H2_RANK_INCREMENT_FRACTION: rank_increment_fraction,
        H2_CONDITIONAL_REPRESENTATION_MULTIPLICITY: conditional_multiplicity,
        H3_CALIBRATED_COST_PER_INDEPENDENT_RELATION: calibrated_per_independent,
        H3_PRIMITIVE_COST_PER_ACCEPTED_RELATION: primitive_per_accepted,
        H3_PRIMITIVE_COST_PER_INDEPENDENT_RELATION: primitive_per_independent,
        H3_WARM_WALL_TIME_PER_INDEPENDENT_RELATION: None,
    }
    censor_reasons = {
        H1_INDEPENDENT_RELATIONS_PER_WORK: "zero calibrated work denominator",
        H1_WORK_TO_FULL_RANK: "full rank not reached",
        H2_CONDITIONAL_REPRESENTATION_MULTIPLICITY: "no decomposed targets",
        H3_CALIBRATED_COST_PER_INDEPENDENT_RELATION: "no independent relations",
        H3_PRIMITIVE_COST_PER_ACCEPTED_RELATION: "no accepted relations",
        H3_PRIMITIVE_COST_PER_INDEPENDENT_RELATION: "no independent relations",
        H3_WARM_WALL_TIME_PER_INDEPENDENT_RELATION: "machine-dependent diagnostic",
    }
    censor_ledger = {
        name: censor_reasons.get(name, "undefined registered metric")
        for name, value in registered_metrics.items()
        if value is None
    }
    metric_status = {
        name: ("censored" if name in censor_ledger else "observed")
        for name in registered_metrics
    }
    metric_status[H3_WARM_WALL_TIME_PER_INDEPENDENT_RELATION] = (
        "censored-machine-dependent"
    )
    return {
        "mode": prepared.base.mode.value,
        "base": {
            "representatives": [_point_json(point) for point in prepared.base.representatives],
            "raw_points": [_point_json(point) for point in prepared.base.raw_points],
            "raw_point_count": raw_count,
            "effective_columns": prepared.base.effective_columns,
            "basis_points": [_point_json(point) for point in prepared.basis_points],
        },
        "preparation": {
            "cost": _cost_dict(prepared.preparation_cost),
            "candidates": prepared.preparation_candidates,
            "charged_count": 1,
        },
        "totals": {
            "cost": _cost_dict(aggregate.total_cost),
            "calibrated_cost": calibrated,
            "primitive_work": primitive_total,
            "candidates": aggregate.total_candidates,
            "targets": aggregate.target_count,
            "accepted_relations": accepted,
        },
        "metrics": metrics,
        "registered_metrics": registered_metrics,
        "metric_status": metric_status,
        "censor_ledger": censor_ledger,
        "equation_audit": equation_rows[:_AUDIT_SAMPLE_LIMIT],
        "target_audit": [
            {
                "index": target["index"], "u": target["u"], "v": target["v"],
                "R": _point_json(target["R"]),
                "search_candidates": result.search_candidates,
                "accepted_relations": len(result.relations),
                "search_cost": _cost_dict(result.search_cost),
            }
            for target, result in zip(targets, results, strict=True)
        ][:_AUDIT_SAMPLE_LIMIT],
    }


def _verify_result(
    prepared: PreparedSearch,
    target: Mapping[str, object],
    result: SearchResult,
    oracle_verify: Callable[..., bool],
) -> None:
    target_point = target["R"]
    u = target["u"]
    v = target["v"]
    if type(u) is not int or type(v) is not int:
        raise ArithmeticError("public target metadata changed type")
    for relation in result.relations:
        if not verify_relation(prepared, relation, target_point):  # type: ignore[arg-type]
            raise ArithmeticError("relation group verification failed")
        if not oracle_verify(prepared.basis_points, relation.normalized_row, u, v):
            raise ArithmeticError("relation oracle verification failed")


def _measure_warm_timings(
    prepared: PreparedSearch,
    targets: tuple[Mapping[str, object], ...],
    oracle_verify: Callable[..., bool],
    independent_relations: int,
) -> dict[str, object]:
    measured_targets = targets[:_AUDIT_SAMPLE_LIMIT]
    totals: list[int] = []
    per_independent: list[float] = []
    for _ in range(_TIMING_REPEATS):
        started = time.perf_counter_ns()
        repeated = tuple(
            prepared.search(target["R"])  # type: ignore[arg-type]
            for target in measured_targets
        )
        elapsed = time.perf_counter_ns() - started
        timing_rows: list[tuple[int, ...]] = []
        for target, result in zip(measured_targets, repeated, strict=True):
            _verify_result(prepared, target, result, oracle_verify)
            v = target["v"]
            if type(v) is not int:
                raise ArithmeticError("public target metadata changed type")
            timing_rows.extend(
                _relation_metric_row(prepared, relation, v)
                for relation in result.relations
            )
        totals.append(elapsed)
        measured_independent = len(canonical_row_space(timing_rows, prepared.fixture.n))
        if measured_independent:
            per_independent.append(elapsed / measured_independent)
    return {
        "machine_dependent": True,
        "repeat_count": _TIMING_REPEATS,
        "target_count": len(measured_targets),
        "full_run_independent_relations": independent_relations,
        "total_ns_samples": totals,
        "per_independent_relation_ns_samples": per_independent,
    }


def _progress_payload(
    key: CellKey,
    manifest_hash: str,
    source_commit: str,
    source_identity: str,
    fixture_resolution_id: str,
    fixture: CurveFixture,
    fixture_attempts: tuple[Mapping[str, object], ...],
    q: Point,
    modes: Mapping[str, object],
    next_target_index: int,
    actual_relation_rows: int,
) -> dict[str, object]:
    return _hashed_payload({
        "schema": 2,
        "cell_id": key.cell_id,
        "cell_key": key.to_dict(),
        "manifest_hash": manifest_hash,
        "source_commit": source_commit,
        "source_identity": source_identity,
        "fixture_resolution_id": fixture_resolution_id,
        "fixture": _fixture_dict(fixture, key.fixture_seed),
        "fixture_attempts": _thaw(fixture_attempts),
        "q": _point_json(q),
        "environment": _environment(),
        "glv_control_passed": True,
        "next_target_index": next_target_index,
        "actual_relation_rows": actual_relation_rows,
        "modes": _thaw(modes),
    })


def run_cell(
    manifest: Manifest,
    key: CellKey,
    manifest_hash: str,
    source_commit: str,
    *,
    source_identity: str | None = None,
    fixture_resolution: FixtureResolution | None = None,
    progress: Callable[[Mapping[str, object]], None] | None = None,
    resumed_payload: Mapping[str, object] | None = None,
    diagnostics: dict[str, object] | None = None,
) -> CellResult:
    if not isinstance(manifest, Manifest):
        raise TypeError("manifest must be a Manifest")
    if not isinstance(key, CellKey):
        raise TypeError("key must be a CellKey")
    _strict_string(manifest_hash, "manifest_hash")
    _strict_string(source_commit, "source_commit")
    source_identity = source_commit if source_identity is None else source_identity
    _strict_string(source_identity, "source_identity")
    if key not in manifest.cell_keys():
        raise ValueError("cell key is not registered by this manifest")
    if progress is not None and not callable(progress):
        raise TypeError("progress must be callable")

    if fixture_resolution is None:
        fixture_resolution = _resolve_fixture_slot(
            key.subgroup_bits, key.fixture_seed
        )
    if not isinstance(fixture_resolution, FixtureResolution):
        raise TypeError("fixture_resolution must be a FixtureResolution")
    if (
        fixture_resolution.subgroup_bits != key.subgroup_bits
        or fixture_resolution.requested_seed != key.fixture_seed
    ):
        raise ValueError("fixture resolution does not match the cell slot")
    fixture = fixture_resolution.fixture
    fixture_attempts = fixture_resolution.attempts
    fixture_resolution_id = fixture_resolution.resolution_id

    bases = build_comparison_bases(
        fixture, key.budget_type, key.factor_base_budget, fixture.seed
    )
    prepared: dict[Mode, PreparedSearch] = {}
    for mode in Mode:
        base_limit = bases[mode].candidate_limit
        candidate_limit = (
            min(base_limit, manifest.candidate_limit)
            if base_limit
            else manifest.candidate_limit
        )
        prepared[mode] = prepare_search(
            fixture, bases[mode], key.arity, candidate_limit
        )
    expanded_rows, compact_rows, orbit_count = _aligned_control_rows(
        prepared[Mode.GLV_EXPANDED], prepared[Mode.MCOLLAPSE_UNWEIGHTED]
    )
    constraint_factors = tuple(
        glv_constraint_rows(
            orbit_count, fixture.eigenvalue, fixture.n, include_rhs=False
        )
    )
    assert_glv_mcollapse_equivalent(
        expanded_rows + constraint_factors,
        compact_rows,
        fixture.eigenvalue,
        fixture.n,
        False,
        orbit_count,
    )
    if not expanded_rows or orbit_count != len(bases[Mode.MCOLLAPSE_UNWEIGHTED].representatives):
        raise ArithmeticError("GLV/MCOLLAPSE control did not prove representative alignment")

    q, targets, oracle_verify = _make_public_targets_and_verifier(fixture, manifest, key)
    next_target_index = 0
    actual_control_rows = 0
    mode_states = {mode: _new_mode_state(prepared[mode]) for mode in Mode}
    if resumed_payload is not None:
        if resumed_payload.get("manifest_hash") != manifest_hash:
            raise ValueError("progress manifest hash mismatch")
        if resumed_payload.get("source_commit") != source_commit:
            raise ValueError("progress source commit mismatch")
        if resumed_payload.get("source_identity") != source_identity:
            raise ValueError("progress source identity mismatch")
        if resumed_payload.get("fixture_resolution_id") != fixture_resolution_id:
            raise ValueError("progress fixture resolution id mismatch")
        if resumed_payload.get("cell_id") != key.cell_id:
            raise ValueError("progress cell ID mismatch")
        _require_payload_hash(resumed_payload, "progress")
        expected_progress_fields = {
            "schema", "cell_id", "cell_key", "manifest_hash", "source_commit",
            "source_identity", "fixture_resolution_id", "fixture",
            "fixture_attempts", "q", "environment", "glv_control_passed", "next_target_index",
            "actual_relation_rows", "modes", "payload_hash",
        }
        if set(resumed_payload) != expected_progress_fields or resumed_payload.get("schema") != 2:
            raise ValueError("progress checkpoint schema mismatch")
        if resumed_payload.get("cell_key") != key.to_dict():
            raise ValueError("progress cell key mismatch")
        if resumed_payload.get("fixture") != _fixture_dict(fixture, key.fixture_seed):
            raise ValueError("progress fixture mismatch")
        if resumed_payload.get("fixture_attempts") != _thaw(fixture_attempts):
            raise ValueError("progress fixture attempts mismatch")
        _validate_environment(resumed_payload.get("environment"))
        if resumed_payload.get("glv_control_passed") is not True:
            raise ValueError("progress GLV control mismatch")
        if _point_from_json(resumed_payload.get("q")) != q:
            raise ValueError("progress public Q mismatch")
        next_target_index = _strict_int(
            resumed_payload.get("next_target_index"),
            "progress next target index",
            minimum=0,
        )
        if next_target_index > len(targets):
            raise ValueError("progress next target index exceeds target count")
        actual_control_rows = _strict_int(
            resumed_payload.get("actual_relation_rows"),
            "progress actual relation rows",
            minimum=0,
        )
        candidate_modes = resumed_payload.get("modes", {})
        if not isinstance(candidate_modes, Mapping) or set(candidate_modes) != {
            mode.value for mode in Mode
        }:
            raise ValueError("progress modes must contain every mode")
        mode_states = {
            mode: _validate_mode_state(
                candidate_modes[mode.value], prepared[mode], targets, next_target_index
            )
            for mode in Mode
        }
        expected_control_rows = (
            mode_states[Mode.GLV_EXPANDED]["accepted_relations"]
            + mode_states[Mode.MCOLLAPSE_UNWEIGHTED]["accepted_relations"]
        )
        if actual_control_rows != expected_control_rows:
            raise ValueError("progress actual relation row count mismatch")

    started_at_zero = next_target_index == 0
    fresh_results: dict[Mode, list[SearchResult]] = {mode: [] for mode in Mode}
    timing_modes: dict[str, object] = {}
    exception_log: list[Mapping[str, object]] = []
    for index in range(next_target_index, len(targets)):
        target = targets[index]
        target_results: dict[Mode, SearchResult] = {}
        for mode in Mode:
            result = prepared[mode].search(target["R"])  # type: ignore[arg-type]
            _verify_result(prepared[mode], target, result, oracle_verify)
            target_results[mode] = result

        expanded_actual, compact_actual, actual_orbit_count = (
            _actual_relation_control_rows(
                prepared[Mode.GLV_EXPANDED],
                target_results[Mode.GLV_EXPANDED],
                prepared[Mode.MCOLLAPSE_UNWEIGHTED],
                target_results[Mode.MCOLLAPSE_UNWEIGHTED],
                target,
            )
        )
        actual_control_rows += len(expanded_actual) + len(compact_actual)
        if any(len(row) != 3 * actual_orbit_count + 2 for row in expanded_actual):
            raise ArithmeticError("GLV/MCOLLAPSE row-space mismatch")
        if any(len(row) != actual_orbit_count + 2 for row in compact_actual):
            raise ArithmeticError("GLV/MCOLLAPSE row-space mismatch")
        constraints = tuple(
            row + (0, 0)
            for row in glv_constraint_rows(
                actual_orbit_count,
                fixture.eigenvalue,
                fixture.n,
                include_rhs=False,
            )
        )
        augmented_expanded = expanded_actual + constraints
        assert_glv_mcollapse_equivalent(
            tuple(row[:-2] for row in augmented_expanded),
            tuple(row[:-2] for row in compact_actual),
            fixture.eigenvalue,
            fixture.n,
            False,
            actual_orbit_count,
        )
        eliminated = eliminate_glv_coefficients(
            tuple(row[:-2] for row in augmented_expanded),
            fixture.eigenvalue,
            fixture.n,
        )
        eliminated_complete = tuple(
            factors + row[-2:]
            for factors, row in zip(eliminated, augmented_expanded, strict=True)
        )
        if canonical_row_space(eliminated_complete, fixture.n) != canonical_row_space(
            compact_actual, fixture.n
        ):
            raise ArithmeticError("GLV/MCOLLAPSE row-space mismatch")

        for mode in Mode:
            result = target_results[mode]
            _update_mode_state(
                mode_states[mode], prepared[mode], target, result,
                manifest.calibration_weights,
            )
            fresh_results[mode].append(result)
        next_target_index = index + 1
        if progress is not None and (
            next_target_index % _CHECKPOINT_BATCH == 0
            or next_target_index == len(targets)
        ):
            progress(
                _progress_payload(
                    key, manifest_hash, source_commit, source_identity,
                    fixture_resolution_id, fixture, fixture_attempts, q,
                    {mode.value: mode_states[mode] for mode in Mode},
                    next_target_index, actual_control_rows,
                )
            )

    if started_at_zero:
        for mode in Mode:
            aggregate_fixture_search(
                prepared[mode], tuple(fresh_results[mode]), manifest.calibration_weights
            )

    completed_modes: dict[str, object] = {
        mode.value: _state_mode_payload(
            prepared[mode], mode_states[mode], manifest.calibration_weights
        )
        for mode in Mode
    }
    equal_primitive_work = min(
        _strict_int(payload["totals"]["primitive_work"], "primitive work", minimum=0)  # type: ignore[index]
        for payload in completed_modes.values()
    )
    for mode in Mode:
        thresholds = mode_states[mode]["rank_work_thresholds"]
        if not isinstance(thresholds, list):
            raise TypeError("mode rank work thresholds must be an array")
        eligible = [work for work in thresholds if work <= equal_primitive_work]
        equal_rank = len(eligible)
        consumed = eligible[-1] if eligible else _primitive_work(prepared[mode].preparation_cost)
        payload = completed_modes[mode.value]
        metrics = payload["metrics"]  # type: ignore[index]
        metrics["equal_primitive_work"] = equal_primitive_work
        metrics["primitive_work_consumed_at_prefix"] = consumed
        metrics["nullity_after_equal_primitive_work"] = max(
            0, bases[mode].effective_columns + 1 - equal_rank
        )
        metrics["nullity_after_equal_candidates"] = max(
            0, bases[mode].effective_columns + 1 - equal_rank
        )
        payload["registered_metrics"][H1_NULLITY_AFTER_EQUAL_CANDIDATES] = metrics[
            "nullity_after_equal_candidates"
        ]
        independent = metrics["independent_relations"]
        if diagnostics is not None:
            timing_modes[mode.value] = _measure_warm_timings(
                prepared[mode], targets, oracle_verify, independent
            )
        for item, reason in payload["censor_ledger"].items():  # type: ignore[index]
            exception_log.append(
                {
                    "category": "censored-metric", "mode": mode.value,
                    "metric": item, "reason": reason,
                }
            )
    if diagnostics is not None:
        diagnostics.clear()
        diagnostics.update(
            {
                "schema": 1,
                "cell_id": key.cell_id,
                "manifest_hash": manifest_hash,
                "source_identity": source_identity,
                "canonical_science_data": False,
                "environment": _environment(),
                "modes": timing_modes,
            }
        )

    short_orbits: list[Mapping[str, object]] = []
    for mode, base in bases.items():
        for representative in base.representatives:
            orbit_size = canonicalize(fixture, representative).orbit_size
            if orbit_size != 6:
                short_orbits.append(
                    {
                        "mode": mode.value,
                        "representative": _point_json(representative),
                        "orbit_size": orbit_size,
                    }
                )
    return CellResult(
        key,
        manifest_hash,
        source_commit,
        source_identity,
        fixture_resolution_id,
        "complete",
        _fixture_dict(fixture, key.fixture_seed),
        q,
        completed_modes,
        True,
        {
            "setup_catalog_rows": (
                len(expanded_rows) + len(compact_rows) + len(constraint_factors)
            ),
            "explicit_constraint_rows": len(constraint_factors),
            "actual_constraint_rows": len(constraint_factors) * len(targets),
            "actual_targets_checked": len(targets),
            "actual_relation_rows": actual_control_rows,
            "candidate_ordering": "aligned-canonical-term-order",
        },
        tuple(exception_log),
        tuple(short_orbits),
        fixture_attempts,
        _environment(),
    )


def _validate_cell_result_payload(result: CellResult) -> None:
    payload = result.to_dict()
    fixture = payload["fixture"]
    if not isinstance(fixture, Mapping) or set(fixture) != {
        "p", "b", "n", "generator", "beta", "eigenvalue", "seed",
        "requested_seed", "accepted_seed",
    }:
        raise ValueError("cell fixture has invalid fields")
    for name in (
        "p", "b", "n", "beta", "eigenvalue", "seed", "requested_seed",
        "accepted_seed",
    ):
        _strict_int(fixture[name], f"cell fixture {name}", minimum=0)
    _point_from_json(fixture["generator"])
    modes = payload["modes"]
    if not isinstance(modes, Mapping) or set(modes) != {mode.value for mode in Mode}:
        raise ValueError("cell modes must contain every registered mode")
    expected_mode_fields = {
        "mode", "base", "preparation", "totals", "metrics",
        "registered_metrics", "metric_status", "censor_ledger",
        "target_audit", "equation_audit",
    }
    target_counts: set[int] = set()
    for mode in Mode:
        value = modes[mode.value]
        if not isinstance(value, Mapping) or set(value) != expected_mode_fields:
            raise ValueError("cell mode payload has invalid fields")
        if value["mode"] != mode.value:
            raise ValueError("cell mode payload label mismatch")
        base = value["base"]
        if not isinstance(base, Mapping) or set(base) != {
            "representatives", "raw_points", "raw_point_count",
            "effective_columns", "basis_points",
        }:
            raise ValueError("cell mode base has invalid fields")
        for name in ("representatives", "raw_points", "basis_points"):
            points = base[name]
            if not isinstance(points, list):
                raise ValueError(f"cell mode base {name} must be an array")
            for point in points:
                _point_from_json(point)
        raw_count = _strict_int(
            base["raw_point_count"], "cell mode raw point count", minimum=0
        )
        effective = _strict_int(
            base["effective_columns"], "cell mode effective columns", minimum=1
        )
        if raw_count != len(base["raw_points"]):
            raise ValueError("cell mode base counts disagree with serialized points")
        preparation = value["preparation"]
        if not isinstance(preparation, Mapping) or set(preparation) != {
            "cost", "candidates", "charged_count"
        }:
            raise ValueError("cell mode preparation has invalid fields")
        preparation_cost = _cost_from(preparation["cost"])
        preparation_candidates = _strict_int(
            preparation["candidates"], "cell mode preparation candidates", minimum=0
        )
        if preparation["charged_count"] != 1:
            raise ValueError("cell mode preparation must be charged exactly once")
        totals = value["totals"]
        if not isinstance(totals, Mapping) or set(totals) != {
            "cost", "calibrated_cost", "primitive_work", "candidates",
            "targets", "accepted_relations",
        }:
            raise ValueError("cell mode totals have invalid fields")
        total_cost = _cost_from(totals["cost"])
        for name in ("primitive_work", "candidates", "targets", "accepted_relations"):
            minimum = 1 if name == "targets" else 0
            _strict_int(totals[name], f"cell mode totals {name}", minimum=minimum)
        target_counts.add(totals["targets"])  # type: ignore[arg-type]
        if totals["primitive_work"] != _primitive_work(total_cost):
            raise ValueError("cell mode primitive work disagrees with total cost")
        if totals["candidates"] < preparation_candidates:
            raise ValueError("cell mode candidates omit preparation")
        if any(
            getattr(total_cost, name) < getattr(preparation_cost, name)
            for name in _COST_NAMES
        ):
            raise ValueError("cell mode total cost omits preparation")
        calibrated = totals["calibrated_cost"]
        if isinstance(calibrated, bool) or not isinstance(calibrated, Real) or not isfinite(calibrated) or calibrated < 0:
            raise ValueError("cell mode totals calibrated cost is invalid")
        registered = value["registered_metrics"]
        if not isinstance(registered, Mapping) or set(registered) != {
            metric for family in REGISTERED_METRIC_FAMILIES.values() for metric in family
        }:
            raise ValueError("cell registered metrics have invalid fields")
        for metric in registered.values():
            if metric is not None and (
                isinstance(metric, bool)
                or not isinstance(metric, Real)
                or not isfinite(metric)
                or metric < 0
            ):
                raise ValueError("cell registered metric value is invalid")
        status = value["metric_status"]
        ledger = value["censor_ledger"]
        if not isinstance(status, Mapping) or set(status) != set(registered):
            raise ValueError("cell metric status has invalid fields")
        if not isinstance(ledger, Mapping) or not set(ledger) <= set(registered):
            raise ValueError("cell censor ledger has invalid fields")
        if any(type(item) is not str or not item for item in status.values()) or any(
            type(item) is not str or not item for item in ledger.values()
        ):
            raise ValueError("cell metric status values are invalid")
        metrics = value["metrics"]
        expected_metrics = {
            "independent_relations", "effective_unknowns",
            "independent_relations_per_unknown", "independent_relations_per_work",
            "work_to_full_rank", "nullity_after_equal_candidates",
            "decomposition_probability", "unique_rows_per_target",
            "rank_increment_fraction", "conditional_representation_multiplicity",
            "primitive_cost_per_accepted_relation",
            "primitive_cost_per_independent_relation",
            "calibrated_cost_per_accepted_relation",
            "calibrated_cost_per_independent_relation", "primitive_scalar_projection",
            "equal_primitive_work", "primitive_work_consumed_at_prefix",
            "nullity_after_equal_primitive_work",
        }
        if not isinstance(metrics, Mapping) or set(metrics) != expected_metrics:
            raise ValueError("cell metrics have invalid fields")
        independent = _strict_int(
            metrics["independent_relations"], "cell independent relations", minimum=0
        )
        unknowns = _strict_int(
            metrics["effective_unknowns"], "cell effective unknowns", minimum=1
        )
        if unknowns != effective + 1 or independent > totals["accepted_relations"]:
            raise ValueError("cell rank metrics disagree with base or relation totals")
        if metrics["nullity_after_equal_candidates"] != metrics[
            "nullity_after_equal_primitive_work"
        ]:
            raise ValueError("cell equal-work nullity metrics disagree")
        target_audit = value["target_audit"]
        equation_audit = value["equation_audit"]
        if not isinstance(target_audit, list) or len(target_audit) != min(
            totals["targets"], _AUDIT_SAMPLE_LIMIT
        ):
            raise ValueError("cell target audit is invalid")
        if not isinstance(equation_audit, list) or len(equation_audit) != min(
            totals["accepted_relations"], _AUDIT_SAMPLE_LIMIT
        ):
            raise ValueError("cell equation audit is invalid")
        for index, audit in enumerate(target_audit):
            if not isinstance(audit, Mapping) or set(audit) != {
                "index", "u", "v", "R", "search_candidates",
                "accepted_relations", "search_cost",
            } or audit["index"] != index:
                raise ValueError("cell target audit index is not monotone")
            _point_from_json(audit["R"])
            _cost_from(audit["search_cost"])
            _strict_int(audit["search_candidates"], "audit search candidates", minimum=0)
            _strict_int(audit["accepted_relations"], "audit accepted relations", minimum=0)
        for equation in equation_audit:
            if not isinstance(equation, Mapping) or set(equation) != {
                "target_index", "coefficients", "q_coefficient", "rhs"
            }:
                raise ValueError("cell equation audit has invalid fields")
            target_index = _strict_int(
                equation["target_index"], "equation target index", minimum=0
            )
            if target_index >= totals["targets"]:
                raise ValueError("cell equation target index exceeds target count")
            coefficients = equation["coefficients"]
            if not isinstance(coefficients, list) or len(coefficients) != len(
                base["basis_points"]
            ) or any(type(item) is not int for item in coefficients):
                raise ValueError("cell equation coefficients are invalid")
    if len(target_counts) != 1:
        raise ValueError("cell mode target totals disagree")
    details = payload["glv_control_details"]
    if not isinstance(details, Mapping) or set(details) != {
        "setup_catalog_rows", "explicit_constraint_rows", "actual_targets_checked",
        "actual_constraint_rows", "actual_relation_rows", "candidate_ordering",
    }:
        raise ValueError("cell GLV control details have invalid fields")
    for name in (
        "setup_catalog_rows", "explicit_constraint_rows", "actual_targets_checked",
        "actual_constraint_rows", "actual_relation_rows",
    ):
        _strict_int(details[name], f"cell GLV control {name}", minimum=0)
    if details["actual_targets_checked"] not in target_counts:
        raise ValueError("cell GLV target count disagrees with mode totals")


def _validate_checkpoint_identity(
    payload: Mapping[str, object],
    key: CellKey,
    manifest_hash: str,
    source_commit: str,
    source_identity: str,
    fixture_resolution_id: str,
) -> None:
    if payload.get("manifest_hash") != manifest_hash:
        raise ValueError("checkpoint manifest hash mismatch")
    if payload.get("source_commit") != source_commit:
        raise ValueError("checkpoint source commit mismatch")
    if payload.get("source_identity") != source_identity:
        raise ValueError("checkpoint source identity mismatch")
    if payload.get("fixture_resolution_id") != fixture_resolution_id:
        raise ValueError("checkpoint fixture resolution id mismatch")
    if payload.get("cell_id") != key.cell_id:
        raise ValueError("checkpoint cell ID mismatch")


def _load_mapping(path: Path) -> Mapping[str, object]:
    value = _read_json(path)
    if not isinstance(value, Mapping):
        raise TypeError(f"JSON root in {path} must be an object")
    return value


def _mode_metric(cell: CellResult, mode: Mode, name: str) -> float | None:
    mode_payload = cell.modes.get(mode.value)
    if not isinstance(mode_payload, Mapping):
        return None
    metrics = mode_payload.get("registered_metrics")
    if not isinstance(metrics, Mapping):
        return None
    value = metrics.get(name)
    if isinstance(value, bool) or not isinstance(value, Real):
        return None
    numeric = float(value)
    return numeric if isfinite(numeric) else None


def _one_sided_sign_p(values: Iterable[float], *, lower_is_better: bool) -> float:
    materialized = tuple(values)
    if not materialized:
        return 1.0
    successes = sum(value < 1.0 if lower_is_better else value > 1.0 for value in materialized)
    return sum(comb(len(materialized), count) for count in range(successes, len(materialized) + 1)) / (
        2 ** len(materialized)
    )


_LOWER_IS_BETTER = frozenset(
    {
        H1_WORK_TO_FULL_RANK,
        H1_NULLITY_AFTER_EQUAL_CANDIDATES,
        H3_CALIBRATED_COST_PER_INDEPENDENT_RELATION,
        H3_PRIMITIVE_COST_PER_ACCEPTED_RELATION,
        H3_PRIMITIVE_COST_PER_INDEPENDENT_RELATION,
        H3_WARM_WALL_TIME_PER_INDEPENDENT_RELATION,
    }
)


def _metric_ratio(
    hypothesis: str, metric: str, mcollapse_value: float, control_value: float
) -> float | None:
    if metric in _LOWER_IS_BETTER and hypothesis != "H3":
        numerator, denominator = control_value, mcollapse_value
    else:
        numerator, denominator = mcollapse_value, control_value
    if denominator == 0:
        return None
    ratio = numerator / denominator
    return ratio if ratio >= 0 and isfinite(ratio) else None


def _metric_inference(
    manifest: Manifest,
    cells: tuple[CellResult, ...],
    hypothesis: str,
    metric: str,
    budget_type: BudgetType,
    mcollapse_mode: Mode,
    control: Mode,
) -> dict[str, object]:
    clusters: dict[str, list[float]] = defaultdict(list)
    bands: dict[int, list[float]] = defaultdict(list)
    ratios: list[float] = []
    censored = 0
    for cell in cells:
        if cell.status != "complete" or cell.key.budget_type is not budget_type:
            continue
        mcollapse_value = _mode_metric(cell, mcollapse_mode, metric)
        control_value = _mode_metric(cell, control, metric)
        if mcollapse_value is None or control_value is None:
            censored += 1
            continue
        ratio = _metric_ratio(hypothesis, metric, mcollapse_value, control_value)
        if ratio is None:
            censored += 1
            continue
        identifier = f"b{cell.key.subgroup_bits}-s{cell.key.fixture_seed}"
        clusters[identifier].append(ratio)
        bands[cell.key.subgroup_bits].append(ratio)
        ratios.append(ratio)
    fixture_ratios = tuple(fmean(values) for values in clusters.values())
    estimate = fmean(fixture_ratios) if fixture_ratios else 1.0
    interval: tuple[float, float] | None = None
    if len(clusters) >= 2:
        interval = clustered_interval(
            clusters, manifest.bootstrap_seed, manifest.bootstrap_iterations
        )
    band_ratios = tuple(
        fmean(bands[bits]) if bands[bits] else estimate
        for bits in manifest.subgroup_bits[:3]
    )
    while len(band_ratios) < 3:
        band_ratios += (estimate,)
    lower_is_better = hypothesis == "H3"
    raw_p = _one_sided_sign_p(
        fixture_ratios, lower_is_better=lower_is_better
    )
    return {
        "estimate": estimate,
        "confidence_interval_95": None if interval is None else list(interval),
        "band_ratios": list(band_ratios),
        "raw_p_value": raw_p,
        "fixture_count": len(clusters),
        "censored_count": censored,
        "status": "observed" if ratios else "censored",
        "direction": "lower-is-better" if metric in _LOWER_IS_BETTER else "higher-is-better",
    }


def _comparison_summary(
    manifest: Manifest,
    cells: tuple[CellResult, ...],
    hypothesis: str,
    budget_type: BudgetType,
    mcollapse_mode: Mode,
    control: Mode,
    partial: bool,
) -> dict[str, object]:
    metric_names = REGISTERED_METRIC_FAMILIES[hypothesis]
    metric_results = {
        metric: _metric_inference(
            manifest, cells, hypothesis, metric, budget_type, mcollapse_mode, control
        )
        for metric in metric_names
    }
    raw_p_values = {
        metric: metric_results[metric]["raw_p_value"] for metric in metric_names
    }
    primary = metric_names[0]
    primary_result = metric_results[primary]
    primary_interval = primary_result["confidence_interval_95"]
    if primary_interval is None:
        primary_interval = [primary_result["estimate"], primary_result["estimate"]]
    positive_floor = 1e-300
    effect = EffectSummary(
        max(primary_result["estimate"], positive_floor),
        tuple(max(value, positive_floor) for value in primary_interval),
        tuple(
            max(value, positive_floor)
            for value in primary_result["band_ratios"]
        ),
    )
    directional = tuple(
        metric
        for metric in metric_names[1:]
        if (
            metric_results[metric]["estimate"] < 1
            if hypothesis == "H3"
            else metric_results[metric]["estimate"] > 1
        )
    )
    evaluation = evaluate_hypothesis(
        hypothesis, effect, raw_p_values, directional
    )
    decision = evaluation.verdict
    complete_controls = all(cell.glv_control_passed for cell in cells)
    if partial or primary_result["fixture_count"] < 2 or not complete_controls:
        decision = "inconclusive"
    return {
        "hypothesis": hypothesis,
        "budget_type": budget_type.value,
        "mcollapse_mode": mcollapse_mode.value,
        "control": control.value,
        "verdict": decision,
        "estimate": primary_result["estimate"],
        "confidence_interval_95": primary_result["confidence_interval_95"],
        "band_ratios": primary_result["band_ratios"],
        "metric_results": metric_results,
        "raw_p_values": raw_p_values,
        "holm_adjusted_p_values": dict(evaluation.adjusted_p_values),
        "fixture_count": primary_result["fixture_count"],
        "censored_count": primary_result["censored_count"],
    }


def _strongest_control_summary(
    hypothesis: str, comparisons: Iterable[Mapping[str, object]]
) -> dict[str, object]:
    items = tuple(comparisons)
    if not items:
        raise ValueError("strongest-control summary requires comparisons")
    strongest = (
        max(items, key=lambda item: item["estimate"])
        if hypothesis == "H3"
        else min(items, key=lambda item: item["estimate"])
    )
    all_supported = all(
        item["verdict"] == "supported in tested regime" for item in items
    )
    if all_supported:
        aggregate_verdict = "supported in tested regime"
    elif strongest["verdict"] == "supported in tested regime":
        aggregate_verdict = "inconclusive"
    else:
        aggregate_verdict = strongest["verdict"]
    result = dict(strongest)
    result.update(
        {
            "hypothesis": hypothesis,
            "strongest_control": strongest["control"],
            "least_favorable_estimate": strongest["estimate"],
            "verdict": aggregate_verdict,
            "supported_against_all_controls": all_supported,
        }
    )
    return result


def _correlations(
    manifest: Manifest, cells: tuple[CellResult, ...]
) -> list[dict[str, object]]:
    fixture_values: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for cell in cells:
        if cell.status != "complete":
            continue
        payload = cell.modes.get(Mode.MCOLLAPSE_WEIGHTED.value)
        if not isinstance(payload, Mapping):
            continue
        base = payload.get("base")
        metrics = payload.get("metrics")
        if not isinstance(base, Mapping) or not isinstance(metrics, Mapping):
            continue
        identifier = f"b{cell.key.subgroup_bits}-s{cell.key.fixture_seed}"
        effective = base.get("effective_columns")
        raw = base.get("raw_point_count")
        if type(effective) is int and effective > 0 and type(raw) is int:
            fixture_values[identifier]["compression"].append(raw / effective)
        for source, destination in (
            ("independent_relations_per_unknown", "yield"),
            ("conditional_representation_multiplicity", "multiplicity"),
            ("rank_increment_fraction", "rank_increment"),
            ("calibrated_cost_per_independent_relation", "calibrated_cost"),
            ("work_to_full_rank", "full_rank_work"),
        ):
            value = metrics.get(source)
            if isinstance(value, Real) and not isinstance(value, bool) and isfinite(value):
                fixture_values[identifier][destination].append(float(value))
    observations: list[FixtureObservation] = []
    for identifier, values in sorted(fixture_values.items()):
        required = ("compression", "yield", "multiplicity", "rank_increment", "calibrated_cost")
        if any(not values[name] for name in required):
            continue
        observations.append(
            FixtureObservation(
                identifier,
                fmean(values["compression"]),
                fmean(values["yield"]),
                fmean(values["multiplicity"]),
                fmean(values["rank_increment"]),
                fmean(values["calibrated_cost"]),
                fmean(values["full_rank_work"]) if values["full_rank_work"] else None,
            )
        )
    unresolved = [
        {
            "association": label,
            "coefficient": None,
            "confidence_interval_95": None,
            "status": "unresolved",
            "fixture_count": len(observations),
            "bootstrap_samples": manifest.bootstrap_iterations,
            "censored_resamples": manifest.bootstrap_iterations,
        }
        for label in _CORRELATION_LABELS
    ]
    if len(observations) < 2:
        return unresolved
    try:
        associations = correlation_report(
            observations, manifest.bootstrap_seed, manifest.bootstrap_iterations
        )
    except ValueError:
        return unresolved
    return [
        {
            "association": association.association,
            "coefficient": association.result.coefficient,
            "confidence_interval_95": list(association.result.interval),
            "status": association.status,
            "fixture_count": association.result.sample_size,
            "bootstrap_samples": association.result.bootstrap_samples,
            "censored_resamples": association.result.censored_resamples,
        }
        for association in associations
    ]


def _build_summary(
    manifest: Manifest,
    manifest_hash: str,
    source_commit: str,
    source_identity: str,
    cells: tuple[CellResult, ...],
    total_cells: int,
    max_cells: int | None,
    failed_fixture_slots: int = 0,
    skipped_cells: int = 0,
) -> dict[str, object]:
    partial = (
        max_cells is not None
        or len(cells) < total_cells
        or failed_fixture_slots > 0
    )
    comparisons = [
        _comparison_summary(
            manifest, cells, hypothesis, budget_type, mcollapse_mode, control, partial
        )
        for hypothesis in ("H1", "H2", "H3")
        for budget_type in manifest.budget_types
        for mcollapse_mode in (Mode.MCOLLAPSE_UNWEIGHTED, Mode.MCOLLAPSE_WEIGHTED)
        for control in (Mode.CLASSIC, Mode.NEGATION, Mode.GLV_EXPANDED)
    ]
    strongest_comparisons = [
        _strongest_control_summary(
            hypothesis,
            tuple(
                item
                for item in comparisons
                if item["hypothesis"] == hypothesis
                and item["budget_type"] == budget_type.value
                and item["mcollapse_mode"] == mcollapse_mode.value
            ),
        )
        for hypothesis in ("H1", "H2", "H3")
        for budget_type in manifest.budget_types
        for mcollapse_mode in (Mode.MCOLLAPSE_UNWEIGHTED, Mode.MCOLLAPSE_WEIGHTED)
    ]
    controls_passed = bool(cells) and failed_fixture_slots == 0 and all(
        cell.status == "complete" and cell.glv_control_passed for cell in cells
    )
    censored_metric_count = sum(
        len(payload.get("censor_ledger", {}))
        for cell in cells
        for payload in cell.modes.values()
        if isinstance(payload, Mapping)
    )
    supports = not partial and controls_passed and any(
        item["verdict"] == "supported in tested regime"
        for item in strongest_comparisons
    )
    return _hashed_payload({
        "schema": 1,
        "manifest_hash": manifest_hash,
        "source_commit": source_commit,
        "source_identity": source_identity,
        "environment": _environment(),
        "completed_cells": len(cells),
        "total_cells": total_cells,
        "partial": partial,
        "max_cells": max_cells,
        "pilot_censored": max_cells is not None,
        "censored_cells": total_cells - len(cells),
        "rejected_cells": sum(cell.status == "rejected" for cell in cells),
        "failed_cells": failed_fixture_slots,
        "failed_fixture_slots": failed_fixture_slots,
        "valid_fixture_shortfall": failed_fixture_slots,
        "skipped_cells": skipped_cells,
        "censored_metric_count": censored_metric_count,
        "short_orbit_count": sum(len(cell.short_orbits) for cell in cells),
        "glv_control": {
            "passed": controls_passed,
            "nonempty_catalog_control": controls_passed,
            "representative_alignment": controls_passed,
            "mismatches": sum(not cell.glv_control_passed for cell in cells),
            "actual_relation_rows": sum(
                cell.glv_control_details.get("actual_relation_rows", 0)
                for cell in cells
            ),
            "actual_targets_checked": sum(
                cell.glv_control_details.get("actual_targets_checked", 0)
                for cell in cells
            ),
        },
        "comparisons": comparisons,
        "strongest_comparisons": strongest_comparisons,
        "correlations": _correlations(manifest, cells),
        "supports_any_hypothesis": supports,
        "h4_h6_tested": False,
    })


def _report(summary: Mapping[str, object]) -> str:
    lines = [
        "# MCOLLAPSE H1-H3 Experiment Report",
        "",
        f"Manifest hash: `{summary['manifest_hash']}`",
        f"Source commit: `{summary['source_commit']}`",
        f"Source identity: `{summary['source_identity']}`",
        f"Completed cells: {summary['completed_cells']}/{summary['total_cells']}",
        f"Partial scheduling: {str(summary['partial']).lower()}",
        f"Pilot max-cells censoring: {str(summary['pilot_censored']).lower()}",
        "",
        "## H1-H3 Verdicts By Budget And Control",
        "",
        "| Hypothesis | Budget | MCOLLAPSE mode | control | Verdict | Estimate | 95% CI | raw p-value | Holm p-value | fixture count | censored |",
        "|---|---|---|---|---|---:|---|---:|---:|---:|---:|",
    ]
    comparisons = summary["comparisons"]
    if not isinstance(comparisons, list):
        raise TypeError("summary comparisons must be a list")
    for item in comparisons:
        if not isinstance(item, Mapping):
            raise TypeError("summary comparison must be a mapping")
        raw = item["raw_p_values"]
        adjusted = item["holm_adjusted_p_values"]
        if not isinstance(raw, Mapping) or not isinstance(adjusted, Mapping):
            raise TypeError("summary p-values must be mappings")
        primary = next(iter(raw))
        interval = item["confidence_interval_95"]
        interval_text = (
            "n/a"
            if interval is None
            else f"[{interval[0]:.6g}, {interval[1]:.6g}]"
        )
        lines.append(
            f"| {item['hypothesis']} | {item['budget_type']} | {item['mcollapse_mode']} | {item['control']} | "
            f"{item['verdict']} | {item['estimate']:.6g} | "
            f"{interval_text} | {raw[primary]:.6g} | "
            f"{adjusted[primary]:.6g} | {item['fixture_count']} | {item['censored_count']} |"
        )
    lines.extend(("", "### Strongest Controls", ""))
    for item in summary["strongest_comparisons"]:
        lines.append(
            f"- {item['hypothesis']} {item['budget_type']} {item['mcollapse_mode']}: "
            f"strongest control `{item['strongest_control']}`, verdict `{item['verdict']}`, "
            f"supported against all controls={str(item['supported_against_all_controls']).lower()}."
        )
    lines.extend(
        (
            "",
            "### Per-Metric Inference",
            "",
            "| Hypothesis | Budget | MCOLLAPSE mode | control | Metric | Estimate | 95% CI | raw p-value | Holm p-value | Status |",
            "|---|---|---|---|---|---:|---|---:|---:|---|",
        )
    )
    for item in comparisons:
        for metric, metric_result in item["metric_results"].items():
            interval = metric_result["confidence_interval_95"]
            interval_text = (
                "n/a"
                if interval is None
                else f"[{interval[0]:.6g}, {interval[1]:.6g}]"
            )
            lines.append(
                f"| {item['hypothesis']} | {item['budget_type']} | {item['mcollapse_mode']} | "
                f"{item['control']} | {metric} | {metric_result['estimate']:.6g} | "
                f"{interval_text} | {metric_result['raw_p_value']:.6g} | "
                f"{item['holm_adjusted_p_values'][metric]:.6g} | {metric_result['status']} |"
            )
    lines.extend(
        (
            "",
            "Holm adjustment applies within each registered H1, H2, or H3 metric family.",
            "A partial schedule cannot produce a positive support verdict.",
            "The primitive scalar projection is the deterministic sum of all CostVector primitive counters; full vectors remain in cell JSON.",
            "Repeated warm wall timing is secondary and censored from scientific verdicts; samples live only in machine-dependent diagnostics.",
            "",
            "## Preregistered Correlations",
            "",
            "Spearman correlation is an association, not a causal claim.",
            "",
            "| Association | Coefficient | 95% CI | Status | fixture count | censored resamples |",
            "|---|---:|---|---|---:|---:|",
        )
    )
    correlations = summary["correlations"]
    if not isinstance(correlations, list):
        raise TypeError("summary correlations must be a list")
    for item in correlations:
        if not isinstance(item, Mapping):
            raise TypeError("summary correlation must be a mapping")
        coefficient = "n/a" if item["coefficient"] is None else f"{item['coefficient']:.6g}"
        interval_value = item["confidence_interval_95"]
        interval = (
            "n/a"
            if interval_value is None
            else f"[{interval_value[0]:.6g}, {interval_value[1]:.6g}]"
        )
        lines.append(
            f"| {item['association']} | {coefficient} | {interval} | {item['status']} | "
            f"{item['fixture_count']} | {item['censored_resamples']} |"
        )
    control = summary["glv_control"]
    if not isinstance(control, Mapping):
        raise TypeError("summary GLV control must be a mapping")
    lines.extend(
        (
            "",
            "## GLV/MCOLLAPSE Algebraic Control",
            "",
            f"GLV/MCOLLAPSE row-space control passed: {str(control['passed']).lower()}.",
            f"Nonempty catalogs and representative alignment: {str(control['nonempty_catalog_control']).lower()}.",
            f"Control mismatches: {control['mismatches']}.",
            f"The actual emitted relation rows checked total {control['actual_relation_rows']} across {control['actual_targets_checked']} targets.",
            "",
            "## Censoring And Rejections",
            "",
            f"Censored cells: {summary['censored_cells']}.",
            f"Censored metric observations: {summary['censored_metric_count']}.",
            f"Rejected fixtures: {summary['rejected_cells']}.",
            f"Failed cells: {summary['failed_cells']}.",
            f"Failed fixture slots: {summary['failed_fixture_slots']}.",
            f"Cells skipped for missing fixture slots: {summary['skipped_cells']}.",
            f"Valid fixture count shortfall: {summary['valid_fixture_shortfall']}.",
            f"Exceptional short orbits: {summary['short_orbit_count']}.",
            "Censored values are represented as null observations, never as zero.",
            "",
            "## Scope",
            "",
            "H4-H6 were not tested. No asymptotic claim follows from this H1-H3 toy-curve run.",
            "",
        )
    )
    return "\n".join(lines)


def run_manifest(
    manifest_path: Path | str,
    output_path: Path | str,
    *,
    resume: bool = False,
    max_cells: int | None = None,
) -> RunResult:
    if type(resume) is not bool:
        raise TypeError("resume must be a boolean")
    if max_cells is not None:
        _strict_int(max_cells, "max_cells", minimum=1)
    manifest_path = Path(manifest_path)
    output = Path(output_path)
    manifest = load_manifest(manifest_path)
    manifest_hash = manifest.canonical_hash
    source_commit = _source_commit()
    source_identity = _source_identity((output,))
    cells_dir = output / "cells"
    progress_dir = cells_dir / ".progress"
    failures_dir = output / "failures"
    diagnostics_dir = output / "diagnostics"
    cells_dir.mkdir(parents=True, exist_ok=True)
    progress_dir.mkdir(parents=True, exist_ok=True)
    failures_dir.mkdir(parents=True, exist_ok=True)
    diagnostics_dir.mkdir(parents=True, exist_ok=True)

    summary_path = output / "summary.json"
    if resume and summary_path.exists():
        previous_summary = _load_mapping(summary_path)
        if previous_summary.get("manifest_hash") != manifest_hash:
            raise ValueError("output manifest hash mismatch")
        if previous_summary.get("source_commit") != source_commit:
            raise ValueError("output source commit mismatch")
        if previous_summary.get("source_identity") != source_identity:
            raise ValueError("output source identity mismatch")
        _require_payload_hash(previous_summary, "output summary")
    atomic_json(output / "manifest.json", manifest.to_dict())

    fixture_resolutions: dict[tuple[int, int], FixtureResolution] = {}
    fixture_failures: dict[tuple[int, int], FixtureAttemptsExhausted] = {}
    for subgroup_bits in manifest.subgroup_bits:
        for requested_seed in manifest.fixture_seeds:
            slot = (subgroup_bits, requested_seed)
            try:
                fixture_resolutions[slot] = _resolve_fixture_slot(*slot)
            except FixtureAttemptsExhausted as error:
                fixture_failures[slot] = error

    all_keys = manifest.cell_keys()
    scheduled_keys = all_keys[:max_cells] if max_cells is not None else all_keys
    skipped_cell_ids: set[str] = set()
    for (subgroup_bits, requested_seed), error in fixture_failures.items():
        dependent = tuple(
            key
            for key in all_keys
            if key.subgroup_bits == subgroup_bits
            and key.fixture_seed == requested_seed
        )
        skipped_cell_ids.update(key.cell_id for key in dependent)
        slot_material = f"{subgroup_bits}:{requested_seed}".encode("ascii")
        slot_id = (
            f"fixture-b{subgroup_bits:02d}-s{requested_seed:08d}-"
            f"{sha256(slot_material).hexdigest()[:12]}"
        )
        _assert_run_identity(
            manifest_path, manifest_hash, source_identity, output
        )
        atomic_json(
            failures_dir / f"{slot_id}.json",
            {
                "fixture_slot_id": slot_id,
                "subgroup_bits": subgroup_bits,
                "requested_fixture_seed": requested_seed,
                "manifest_hash": manifest_hash,
                "source_commit": source_commit,
                "source_identity": source_identity,
                "category": "fixture-attempts-exhausted",
                "message": str(error),
                "fixture_attempts": _thaw(error.attempts),
                "valid_fixture_shortfall": 1,
                "dependent_cells": [
                    {
                        "cell_id": key.cell_id,
                        "cell_key": key.to_dict(),
                        "status": "skipped-fixture-slot",
                    }
                    for key in dependent
                ],
            },
        )

    completed: list[CellResult] = []
    for key in scheduled_keys:
        slot = (key.subgroup_bits, key.fixture_seed)
        if slot in fixture_failures:
            continue
        fixture_resolution = fixture_resolutions[slot]
        cell_path = cells_dir / f"{key.cell_id}.json"
        progress_path = progress_dir / f"{key.cell_id}.json"
        if resume and cell_path.exists():
            payload = _load_mapping(cell_path)
            _validate_checkpoint_identity(
                payload,
                key,
                manifest_hash,
                source_commit,
                source_identity,
                fixture_resolution.resolution_id,
            )
            completed.append(CellResult.from_mapping(payload))
            continue

        resumed_payload: Mapping[str, object] | None = None
        if resume and progress_path.exists():
            resumed_payload = _load_mapping(progress_path)
            _validate_checkpoint_identity(
                resumed_payload,
                key,
                manifest_hash,
                source_commit,
                source_identity,
                fixture_resolution.resolution_id,
            )

        def checkpoint(payload: Mapping[str, object]) -> None:
            _assert_run_identity(
                manifest_path, manifest_hash, source_identity, output
            )
            atomic_json(progress_path, payload)

        diagnostic_payload: dict[str, object] = {}
        try:
            cell = run_cell(
                manifest,
                key,
                manifest_hash,
                source_commit,
                source_identity=source_identity,
                fixture_resolution=fixture_resolution,
                progress=checkpoint,
                resumed_payload=resumed_payload,
                diagnostics=diagnostic_payload,
            )
        except (ArithmeticError, RuntimeError, TypeError, ValueError) as error:
            message = str(error)
            if message == "GLV/MCOLLAPSE row-space mismatch":
                category = "control-failure"
            elif "group verification" in message:
                category = "group-verification-failure"
            elif "oracle verification" in message:
                category = "oracle-verification-failure"
            elif "relation" in message:
                category = "relation-verification-failure"
            else:
                category = "cell-execution-failure"
            atomic_json(
                failures_dir / f"{key.cell_id}.json",
                {
                    "cell_id": key.cell_id,
                    "cell_key": key.to_dict(),
                    "manifest_hash": manifest_hash,
                    "source_commit": source_commit,
                    "source_identity": source_identity,
                    "category": category,
                    "message": message,
                },
            )
            raise
        _assert_run_identity(manifest_path, manifest_hash, source_identity, output)
        atomic_json(
            diagnostics_dir / f"{key.cell_id}.timing.json", diagnostic_payload
        )
        _assert_run_identity(manifest_path, manifest_hash, source_identity, output)
        atomic_json(cell_path, cell.to_dict())
        if progress_path.exists():
            progress_path.unlink()
        completed.append(cell)

    summary = _build_summary(
        manifest,
        manifest_hash,
        source_commit,
        source_identity,
        tuple(completed),
        len(all_keys),
        max_cells,
        len(fixture_failures),
        len(skipped_cell_ids),
    )
    _assert_run_identity(manifest_path, manifest_hash, source_identity, output)
    atomic_json(summary_path, summary)
    _assert_run_identity(manifest_path, manifest_hash, source_identity, output)
    _atomic_text(output / "report.md", _report(summary))
    return RunResult(
        manifest_hash,
        source_commit,
        len(completed),
        len(all_keys),
        bool(summary["partial"]),
        summary,
    )
