"""Fixture-level statistical summaries for the registered MCOLLAPSE hypotheses."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite, sqrt
from numbers import Real
from random import Random
from statistics import fmean
from types import MappingProxyType
from typing import Callable, Iterable, Mapping, Sequence

from mcollapse.costs import CostVector
from mcollapse.relations import (
    PreparedSearch,
    SearchResult,
    aggregate_candidates,
    aggregate_cost,
)


VERDICTS = (
    "supported in tested regime",
    "rejected in tested regime",
    "inconclusive",
)
MIN_DEFINED_BOOTSTRAP_FRACTION = 0.95
MULTIPLICITY_STATEMENT = (
    "Multiplicity-adjusted support decision (Holm); "
    "percentile confidence interval is unadjusted"
)
H1_INDEPENDENT_RELATIONS_PER_UNKNOWN = "independent-relations-per-unknown"
H1_INDEPENDENT_RELATIONS_PER_WORK = "independent-relations-per-work"
H1_WORK_TO_FULL_RANK = "work-to-full-rank"
H1_NULLITY_AFTER_EQUAL_CANDIDATES = "nullity-after-equal-candidates"
H2_DECOMPOSITION_PROBABILITY = "decomposition-probability"
H2_UNIQUE_NORMALIZED_VECTORS_PER_TARGET = "unique-normalized-vectors-per-target"
H2_RANK_INCREMENT_FRACTION = "rank-increment-fraction"
H2_CONDITIONAL_REPRESENTATION_MULTIPLICITY = (
    "conditional-representation-multiplicity"
)
H3_CALIBRATED_COST_PER_INDEPENDENT_RELATION = (
    "calibrated-cost-per-independent-relation"
)
H3_PRIMITIVE_COST_PER_ACCEPTED_RELATION = "primitive-cost-per-accepted-relation"
H3_PRIMITIVE_COST_PER_INDEPENDENT_RELATION = (
    "primitive-cost-per-independent-relation"
)
H3_WARM_WALL_TIME_PER_INDEPENDENT_RELATION = (
    "warm-wall-time-per-independent-relation"
)
REGISTERED_METRIC_FAMILIES: Mapping[str, tuple[str, ...]] = MappingProxyType(
    {
        "H1": (
            H1_INDEPENDENT_RELATIONS_PER_UNKNOWN,
            H1_INDEPENDENT_RELATIONS_PER_WORK,
            H1_WORK_TO_FULL_RANK,
            H1_NULLITY_AFTER_EQUAL_CANDIDATES,
        ),
        "H2": (
            H2_DECOMPOSITION_PROBABILITY,
            H2_UNIQUE_NORMALIZED_VECTORS_PER_TARGET,
            H2_RANK_INCREMENT_FRACTION,
            H2_CONDITIONAL_REPRESENTATION_MULTIPLICITY,
        ),
        "H3": (
            H3_CALIBRATED_COST_PER_INDEPENDENT_RELATION,
            H3_PRIMITIVE_COST_PER_ACCEPTED_RELATION,
            H3_PRIMITIVE_COST_PER_INDEPENDENT_RELATION,
            H3_WARM_WALL_TIME_PER_INDEPENDENT_RELATION,
        ),
    }
)


def _finite_number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be numeric")
    result = float(value)
    if not isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _finite_values(values: Iterable[Real], name: str) -> tuple[float, ...]:
    try:
        materialized = tuple(values)
    except TypeError as error:
        raise TypeError(f"{name} must be iterable") from error
    return tuple(_finite_number(value, name) for value in materialized)


def _positive_integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def _nonnegative_integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < 0:
        raise ValueError(f"{name} must be nonnegative")
    return value


def _fixture_id(value: object) -> str | int:
    if type(value) is int:
        return value
    if type(value) is str:
        if not value:
            raise ValueError("fixture ID must not be empty")
        return value
    raise TypeError("fixture ID must be a string or integer")


def _fixture_sort_key(value: object) -> tuple[str, str | int]:
    identifier = _fixture_id(value)
    return type(identifier).__name__, identifier


def _metric_name(value: object, name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    if not value or value != value.strip():
        raise ValueError(f"{name} must be nonempty and trimmed")
    return value


def _p_value(value: object, name: str) -> float:
    result = _finite_number(value, name)
    if not 0 <= result <= 1:
        raise ValueError(f"{name} must be in [0, 1]")
    return result


@dataclass(frozen=True)
class FixtureObservation:
    """One independent fixture record; target rows are intentionally absent."""

    fixture_id: str | int
    orbit_compression: float
    independent_relation_yield: float
    normalized_representation_multiplicity: float
    rank_increment_fraction: float
    calibrated_cost: float
    full_rank_work: float | None = None

    def __post_init__(self) -> None:
        _fixture_id(self.fixture_id)
        for name in (
            "orbit_compression",
            "independent_relation_yield",
            "normalized_representation_multiplicity",
            "calibrated_cost",
        ):
            value = _finite_number(getattr(self, name), name)
            if value < 0:
                raise ValueError(f"{name} must be nonnegative")
            object.__setattr__(self, name, value)
        fraction = _finite_number(self.rank_increment_fraction, "rank_increment_fraction")
        if not 0 <= fraction <= 1:
            raise ValueError("rank_increment_fraction must be in [0, 1]")
        object.__setattr__(self, "rank_increment_fraction", fraction)
        if self.full_rank_work is not None:
            work = _finite_number(self.full_rank_work, "full_rank_work")
            if work <= 0:
                raise ValueError("full_rank_work must be positive when observed")
            object.__setattr__(self, "full_rank_work", work)


@dataclass(frozen=True)
class EffectSummary:
    estimate: float
    interval: tuple[float, float]
    band_ratios: tuple[float, float, float]

    def __post_init__(self) -> None:
        estimate, interval, bands = _validated_effect(
            self.estimate, self.interval, self.band_ratios
        )
        object.__setattr__(self, "estimate", estimate)
        object.__setattr__(self, "interval", interval)
        object.__setattr__(self, "band_ratios", bands)


@dataclass(frozen=True)
class CorrelationInterval:
    coefficient: float
    interval: tuple[float, float]
    sample_size: int
    bootstrap_samples: int
    censored_resamples: int

    def __post_init__(self) -> None:
        coefficient = _finite_number(self.coefficient, "correlation coefficient")
        if not -1 <= coefficient <= 1:
            raise ValueError("correlation coefficient must be in [-1, 1]")
        if not isinstance(self.interval, tuple):
            raise TypeError("correlation interval must be a tuple")
        if len(self.interval) != 2:
            raise ValueError("correlation interval must contain exactly two bounds")
        low, high = (
            _finite_number(value, "correlation interval bound")
            for value in self.interval
        )
        if not -1 <= low <= high <= 1:
            raise ValueError("correlation interval must be ordered within [-1, 1]")
        sample_size = _nonnegative_integer(self.sample_size, "sample_size")
        if sample_size < 2:
            raise ValueError("sample_size must be at least two")
        bootstrap_samples = _positive_integer(
            self.bootstrap_samples, "bootstrap_samples"
        )
        censored_resamples = _nonnegative_integer(
            self.censored_resamples, "censored_resamples"
        )
        if censored_resamples >= bootstrap_samples:
            raise ValueError("at least one bootstrap resample must be defined")
        object.__setattr__(self, "coefficient", coefficient)
        object.__setattr__(self, "interval", (low, high))
        object.__setattr__(self, "sample_size", sample_size)
        object.__setattr__(self, "bootstrap_samples", bootstrap_samples)
        object.__setattr__(self, "censored_resamples", censored_resamples)

    @property
    def defined_resamples(self) -> int:
        return self.bootstrap_samples - self.censored_resamples

    @property
    def defined_fraction(self) -> float:
        return self.defined_resamples / self.bootstrap_samples

    @property
    def interpretable(self) -> bool:
        return self.defined_fraction >= MIN_DEFINED_BOOTSTRAP_FRACTION

    @property
    def status(self) -> str:
        if not self.interpretable:
            return "unresolved"
        low, high = self.interval
        if low > 0:
            return "positive"
        if high < 0:
            return "negative"
        return "unresolved"


@dataclass(frozen=True)
class CorrelationAssociation:
    association: str
    result: CorrelationInterval

    def __post_init__(self) -> None:
        if not isinstance(self.association, str):
            raise TypeError("association label must be a string")
        if (
            not self.association
            or self.association != self.association.strip()
            or " vs " not in self.association
        ):
            raise ValueError("association label must be a trimmed 'left vs right' label")
        if "caus" in self.association.casefold():
            raise ValueError("association labels must not claim causality")
        if not isinstance(self.result, CorrelationInterval):
            raise TypeError("result must be a CorrelationInterval")

    @property
    def status(self) -> str:
        return self.result.status


@dataclass(frozen=True)
class HypothesisEvaluation:
    verdict: str
    primary_metric: str
    raw_primary_p_value: float
    adjusted_primary_p_value: float
    adjusted_p_values: tuple[tuple[str, float], ...]
    supporting_secondary_metrics: tuple[str, ...]
    multiplicity_statement: str = MULTIPLICITY_STATEMENT

    def __post_init__(self) -> None:
        if self.verdict not in VERDICTS:
            raise ValueError("verdict must be a registered verdict")
        primary_metric = _metric_name(self.primary_metric, "primary_metric")
        raw_primary = _p_value(self.raw_primary_p_value, "raw_primary_p_value")
        adjusted_primary = _p_value(
            self.adjusted_primary_p_value, "adjusted_primary_p_value"
        )
        if adjusted_primary < raw_primary:
            raise ValueError("adjusted primary p-value cannot be below its raw value")
        if not isinstance(self.adjusted_p_values, tuple):
            raise TypeError("adjusted_p_values must be a tuple")
        adjusted: list[tuple[str, float]] = []
        for item in self.adjusted_p_values:
            if not isinstance(item, tuple) or len(item) != 2:
                raise TypeError("adjusted_p_values entries must be name-value tuples")
            adjusted.append(
                (_metric_name(item[0], "metric name"), _p_value(item[1], "adjusted p-value"))
            )
        adjusted_tuple = tuple(adjusted)
        adjusted_names = tuple(name for name, _ in adjusted_tuple)
        if len(adjusted_names) != len(set(adjusted_names)):
            raise ValueError("adjusted metric names must be unique")
        adjusted_by_name = dict(adjusted_tuple)
        if primary_metric not in adjusted_by_name:
            raise ValueError("adjusted_p_values must include the primary metric")
        if adjusted_by_name[primary_metric] != adjusted_primary:
            raise ValueError("adjusted primary p-value must match its named value")
        if self.verdict == VERDICTS[0] and adjusted_primary > 0.05:
            raise ValueError("support requires an adjusted primary p-value at or below 0.05")
        if not isinstance(self.supporting_secondary_metrics, tuple):
            raise TypeError("supporting_secondary_metrics must be a tuple")
        names = self.supporting_secondary_metrics
        for name in names:
            _metric_name(name, "supporting secondary metric name")
        if len(names) != len(set(names)):
            raise ValueError("supporting secondary metric names must be unique")
        if primary_metric in names or any(name not in adjusted_by_name for name in names):
            raise ValueError("supporting secondary metrics must name non-primary family metrics")
        if any(adjusted_by_name[name] > 0.05 for name in names):
            raise ValueError("supporting secondary metrics must pass the Holm decision threshold")
        if self.multiplicity_statement != MULTIPLICITY_STATEMENT:
            raise ValueError("multiplicity statement must describe the Holm support decision")
        object.__setattr__(self, "primary_metric", primary_metric)
        object.__setattr__(self, "raw_primary_p_value", raw_primary)
        object.__setattr__(self, "adjusted_primary_p_value", adjusted_primary)
        object.__setattr__(self, "adjusted_p_values", adjusted_tuple)


@dataclass(frozen=True)
class FixtureSearchMetrics:
    total_cost: CostVector
    total_candidates: int
    target_count: int
    accepted_relations: int
    calibrated_cost: float | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.total_cost, CostVector):
            raise TypeError("total_cost must be a CostVector")
        for name in ("total_candidates", "target_count", "accepted_relations"):
            object.__setattr__(
                self, name, _nonnegative_integer(getattr(self, name), name)
            )
        if self.calibrated_cost is not None:
            calibrated = _finite_number(self.calibrated_cost, "calibrated_cost")
            if calibrated < 0:
                raise ValueError("calibrated_cost must be nonnegative")
            object.__setattr__(self, "calibrated_cost", calibrated)


@dataclass(frozen=True)
class CensoredMetricSummary:
    values: tuple[float, ...]
    observed_count: int
    censored_count: int

    def __post_init__(self) -> None:
        if not isinstance(self.values, tuple):
            raise TypeError("values must be a tuple")
        values = _finite_values(self.values, "observed metric values")
        if any(value <= 0 for value in values):
            raise ValueError("observed metric values must be positive")
        observed = _nonnegative_integer(self.observed_count, "observed_count")
        censored = _nonnegative_integer(self.censored_count, "censored_count")
        if observed != len(values):
            raise ValueError("observed_count must equal the number of values")
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "observed_count", observed)
        object.__setattr__(self, "censored_count", censored)


def percentile(values: Iterable[Real], probability: Real) -> float:
    """Return a linearly interpolated percentile over finite numbers."""

    ordered = sorted(_finite_values(values, "percentile values"))
    if not ordered:
        raise ValueError("percentile values must not be empty")
    probability_value = _finite_number(probability, "probability")
    if not 0 <= probability_value <= 1:
        raise ValueError("probability must be in [0, 1]")
    position = (len(ordered) - 1) * probability_value
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def clustered_interval(
    clusters: Mapping[object, Iterable[Real]],
    seed: int,
    iterations: int,
    statistic: Callable[[Sequence[float]], Real] = fmean,
) -> tuple[float, float]:
    """Bootstrap fixture summaries, giving every fixture one sampling unit."""

    if not isinstance(clusters, Mapping):
        raise TypeError("clusters must be a mapping of fixture IDs to values")
    if len(clusters) < 2:
        raise ValueError("at least two fixture clusters are required")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer")
    iterations = _positive_integer(iterations, "iterations")
    if not callable(statistic):
        raise TypeError("statistic must be callable")

    identifiers = tuple(clusters)
    for identifier in identifiers:
        _fixture_id(identifier)
    identifiers = tuple(sorted(identifiers, key=_fixture_sort_key))
    fixture_values: dict[object, float] = {}
    for identifier in identifiers:
        values = _finite_values(clusters[identifier], "cluster values")
        if not values:
            raise ValueError("every fixture cluster must be nonempty")
        fixture_values[identifier] = fmean(values)

    rng = Random(seed)
    samples: list[float] = []
    for _ in range(iterations):
        sampled = tuple(
            fixture_values[rng.choice(identifiers)] for _ in identifiers
        )
        samples.append(_finite_number(statistic(sampled), "bootstrap statistic"))
    return percentile(samples, 0.025), percentile(samples, 0.975)


def average_ranks(values: Iterable[Real]) -> tuple[float, ...]:
    finite = _finite_values(values, "rank values")
    if not finite:
        raise ValueError("rank values must not be empty")
    ordered = sorted(enumerate(finite), key=lambda item: item[1])
    ranks = [0.0] * len(ordered)
    start = 0
    while start < len(ordered):
        end = start + 1
        while end < len(ordered) and ordered[end][1] == ordered[start][1]:
            end += 1
        rank = (start + 1 + end) / 2
        for index, _ in ordered[start:end]:
            ranks[index] = rank
        start = end
    return tuple(ranks)


def pearson(left: Iterable[Real], right: Iterable[Real]) -> float:
    x = _finite_values(left, "left correlation values")
    y = _finite_values(right, "right correlation values")
    if len(x) != len(y) or len(x) < 2:
        raise ValueError("correlation inputs require equal lengths of at least two")
    x_mean = fmean(x)
    y_mean = fmean(y)
    numerator = sum(
        (a - x_mean) * (b - y_mean) for a, b in zip(x, y, strict=True)
    )
    x_variance = sum((value - x_mean) ** 2 for value in x)
    y_variance = sum((value - y_mean) ** 2 for value in y)
    denominator = sqrt(x_variance * y_variance)
    if denominator == 0:
        raise ValueError("correlation is undefined for a constant input")
    coefficient = numerator / denominator
    if coefficient >= 1.0:
        return 1.0
    if coefficient <= -1.0:
        return -1.0
    return coefficient


def spearman(left: Iterable[Real], right: Iterable[Real]) -> float:
    x = _finite_values(left, "left Spearman values")
    y = _finite_values(right, "right Spearman values")
    if len(x) != len(y) or len(x) < 2:
        raise ValueError("Spearman inputs require equal lengths of at least two")
    return pearson(average_ranks(x), average_ranks(y))


def _fixture_pairs(
    records: Mapping[object, tuple[Real, Real]],
) -> tuple[tuple[object, ...], dict[object, tuple[float, float]]]:
    if not isinstance(records, Mapping):
        raise TypeError("records must map fixture IDs to one numeric pair")
    if len(records) < 2:
        raise ValueError("at least two fixture records are required")
    identifiers = tuple(records)
    for identifier in identifiers:
        _fixture_id(identifier)
    identifiers = tuple(sorted(identifiers, key=_fixture_sort_key))
    pairs: dict[object, tuple[float, float]] = {}
    for identifier in identifiers:
        pair = records[identifier]
        if not isinstance(pair, tuple) or len(pair) != 2:
            raise ValueError("each fixture must have exactly one numeric pair")
        pairs[identifier] = (
            _finite_number(pair[0], "left fixture value"),
            _finite_number(pair[1], "right fixture value"),
        )
    return identifiers, pairs


def clustered_spearman(
    records: Mapping[object, tuple[Real, Real]], seed: int, iterations: int
) -> CorrelationInterval:
    """Bootstrap one pair per fixture and count undefined resamples as censored."""

    identifiers, pairs = _fixture_pairs(records)
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer")
    iterations = _positive_integer(iterations, "iterations")
    coefficient = spearman(
        (pairs[item][0] for item in identifiers),
        (pairs[item][1] for item in identifiers),
    )
    rng = Random(seed)
    samples: list[float] = []
    censored = 0
    for _ in range(iterations):
        chosen = tuple(rng.choice(identifiers) for _ in identifiers)
        try:
            samples.append(
                spearman(
                    (pairs[item][0] for item in chosen),
                    (pairs[item][1] for item in chosen),
                )
            )
        except ValueError:
            censored += 1
    if not samples:
        raise ValueError("every bootstrap correlation resample was undefined")
    return CorrelationInterval(
        coefficient,
        (percentile(samples, 0.025), percentile(samples, 0.975)),
        len(identifiers),
        iterations,
        censored,
    )


def _fixture_observations(
    observations: Iterable[FixtureObservation],
) -> tuple[FixtureObservation, ...]:
    try:
        records = tuple(observations)
    except TypeError as error:
        raise TypeError("observations must be iterable") from error
    if not all(isinstance(record, FixtureObservation) for record in records):
        raise TypeError("observations must contain only FixtureObservation values")
    identifiers = tuple(record.fixture_id for record in records)
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("fixture observations must have unique IDs")
    return records


def correlation_report(
    observations: Iterable[FixtureObservation], seed: int, iterations: int
) -> tuple[CorrelationAssociation, ...]:
    """Compute the three preregistered fixture-level associations."""

    records = _fixture_observations(observations)
    specifications = (
        (
            "orbit compression vs independent-relation yield",
            "orbit_compression",
            "independent_relation_yield",
        ),
        (
            "normalized representation multiplicity vs rank-increment fraction",
            "normalized_representation_multiplicity",
            "rank_increment_fraction",
        ),
        (
            "calibrated cost vs independent-relation yield",
            "calibrated_cost",
            "independent_relation_yield",
        ),
    )
    return tuple(
        CorrelationAssociation(
            association,
            clustered_spearman(
                {
                    record.fixture_id: (
                        getattr(record, left_name),
                        getattr(record, right_name),
                    )
                    for record in records
                },
                seed,
                iterations,
            ),
        )
        for association, left_name, right_name in specifications
    )


def holm_adjust(p_values: Iterable[Real]) -> tuple[float, ...]:
    values = _finite_values(p_values, "p-values")
    if any(value < 0 or value > 1 for value in values):
        raise ValueError("p-values must be in [0, 1]")
    ordered = sorted(enumerate(values), key=lambda item: item[1])
    adjusted = [0.0] * len(values)
    previous = 0.0
    for rank, (index, value) in enumerate(ordered):
        corrected = min(1.0, (len(values) - rank) * value)
        previous = max(previous, corrected)
        adjusted[index] = previous
    return tuple(adjusted)


def _validated_effect(
    estimate: Real,
    interval: Sequence[Real],
    band_effects: Sequence[Real],
) -> tuple[float, tuple[float, float], tuple[float, float, float]]:
    estimate_value = _finite_number(estimate, "estimate")
    try:
        interval_values = tuple(interval)
        bands_values = tuple(band_effects)
    except TypeError as error:
        raise ValueError("interval and band ratios must be sequences") from error
    if len(interval_values) != 2:
        raise ValueError("interval must contain exactly two bounds")
    if len(bands_values) != 3:
        raise ValueError("exactly three band ratios are required")
    low, high = (_finite_number(value, "interval bound") for value in interval_values)
    bands = tuple(_finite_number(value, "band ratio") for value in bands_values)
    if estimate_value <= 0 or low <= 0 or high <= 0 or any(value <= 0 for value in bands):
        raise ValueError("effect ratios must be positive")
    if low > high:
        raise ValueError("interval bounds must be ordered")
    return estimate_value, (low, high), bands  # type: ignore[return-value]


def verdict(
    name: str,
    estimate: Real,
    interval: Sequence[Real],
    band_effects: Sequence[Real],
) -> str:
    if name not in ("H1", "H2", "H3"):
        raise ValueError("unknown hypothesis")
    estimate_value, (low, high), bands = _validated_effect(
        estimate, interval, band_effects
    )
    if name == "H3":
        if estimate_value <= 0.90 and high < 0.90 and all(value < 1.0 for value in bands):
            return VERDICTS[0]
        if low >= 1.0:
            return VERDICTS[1]
        return VERDICTS[2]
    if estimate_value >= 1.10 and low > 1.0 and all(value > 1.0 for value in bands):
        return VERDICTS[0]
    if high <= 1.0:
        return VERDICTS[1]
    return VERDICTS[2]


def evaluate_hypothesis(
    name: str,
    summary: EffectSummary,
    metric_p_values: Mapping[str, Real],
    direction_supporting_metrics: Iterable[str] = (),
) -> HypothesisEvaluation:
    if not isinstance(summary, EffectSummary):
        raise TypeError("summary must be an EffectSummary")
    try:
        metric_names = REGISTERED_METRIC_FAMILIES[name]
    except (KeyError, TypeError) as error:
        raise ValueError("unknown hypothesis") from error
    primary_metric = metric_names[0]
    if not isinstance(metric_p_values, Mapping):
        raise TypeError("metric_p_values must be a mapping")
    p_values = {
        _metric_name(metric, "metric name"): _p_value(p_value, "metric p-value")
        for metric, p_value in metric_p_values.items()
    }
    if set(p_values) != set(metric_names):
        raise ValueError("metric_p_values must exactly match the registered metric family")
    adjusted_values = holm_adjust(p_values[metric] for metric in metric_names)
    adjusted = tuple(zip(metric_names, adjusted_values, strict=True))
    adjusted_by_name = dict(adjusted)
    try:
        directional = tuple(direction_supporting_metrics)
    except TypeError as error:
        raise TypeError("direction_supporting_metrics must be iterable") from error
    for metric in directional:
        _metric_name(metric, "direction-supporting metric name")
    if len(directional) != len(set(directional)):
        raise ValueError("direction-supporting metric names must be unique")
    if primary_metric in directional or any(metric not in p_values for metric in directional):
        raise ValueError("direction-supporting metrics must name non-primary family metrics")
    supporting = tuple(
        metric for metric in directional if adjusted_by_name[metric] <= 0.05
    )
    primary_verdict = verdict(
        name, summary.estimate, summary.interval, summary.band_ratios
    )
    adjusted_primary = adjusted_by_name[primary_metric]
    if primary_verdict == VERDICTS[0] and adjusted_primary > 0.05:
        primary_verdict = VERDICTS[2]
    return HypothesisEvaluation(
        primary_verdict,
        primary_metric,
        p_values[primary_metric],
        adjusted_primary,
        adjusted,
        supporting,
    )


def aggregate_fixture_search(
    prepared: PreparedSearch,
    results: Iterable[SearchResult],
    calibration_weights: Mapping[str, Real] | None = None,
) -> FixtureSearchMetrics:
    """Aggregate one prepared configuration and all of its target searches."""

    try:
        target_results = tuple(results)
    except TypeError as error:
        raise TypeError("results must be iterable") from error
    total_cost = aggregate_cost(prepared, target_results)
    total_candidates = aggregate_candidates(prepared, target_results)
    calibrated = (
        None if calibration_weights is None else total_cost.calibrated(calibration_weights)
    )
    return FixtureSearchMetrics(
        total_cost,
        total_candidates,
        len(target_results),
        sum(len(result.relations) for result in target_results),
        calibrated,
    )


def summarize_full_rank_work(
    observations: Iterable[FixtureObservation],
) -> CensoredMetricSummary:
    records = _fixture_observations(observations)
    values = tuple(
        record.full_rank_work
        for record in records
        if record.full_rank_work is not None
    )
    return CensoredMetricSummary(values, len(values), len(records) - len(values))
