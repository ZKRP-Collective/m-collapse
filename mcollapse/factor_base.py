from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from hashlib import sha256
from typing import Iterable

from mcollapse.curves import CurveFixture, Point, validate_fixture
from mcollapse.orbits import canonicalize, orbit_points


class Mode(str, Enum):
    CLASSIC = "classic"
    NEGATION = "negation"
    GLV_EXPANDED = "glv-expanded"
    MCOLLAPSE_UNWEIGHTED = "mcollapse-unweighted"
    MCOLLAPSE_WEIGHTED = "mcollapse-weighted"


class BudgetType(str, Enum):
    EQUAL_RAW = "equal-raw"
    EQUAL_UNKNOWNS = "equal-unknowns"
    EQUAL_WORK = "equal-work"


def _is_affine_point(point: Point) -> bool:
    return (
        isinstance(point, tuple)
        and len(point) == 2
        and all(isinstance(coordinate, int) for coordinate in point)
    )


@dataclass(frozen=True)
class FactorBase:
    mode: Mode
    representatives: tuple[Point, ...]
    raw_points: tuple[Point, ...]
    effective_columns: int
    candidate_limit: int

    def __post_init__(self) -> None:
        if not isinstance(self.mode, Mode):
            raise TypeError("mode must be a Mode")
        if not isinstance(self.representatives, tuple) or not self.representatives:
            raise ValueError("representatives must be a nonempty tuple")
        if not isinstance(self.raw_points, tuple) or not self.raw_points:
            raise ValueError("raw_points must be a nonempty tuple")
        if not all(_is_affine_point(point) for point in self.representatives):
            raise ValueError("representatives must contain only affine points")
        if not all(_is_affine_point(point) for point in self.raw_points):
            raise ValueError("raw_points must contain only affine points")
        if len(set(self.representatives)) != len(self.representatives):
            raise ValueError("representatives must be unique")
        if len(set(self.raw_points)) != len(self.raw_points):
            raise ValueError("raw_points must be unique")
        if (
            type(self.effective_columns) is not int
            or self.effective_columns != len(self.representatives)
        ):
            raise ValueError("effective_columns must equal the representative count")
        if type(self.candidate_limit) is not int or self.candidate_limit < 0:
            raise ValueError("candidate_limit must be a nonnegative integer")

    def allowed_points(
        self, fixture: CurveFixture, representative: Point
    ) -> tuple[tuple[int, int], ...]:
        """Return concrete points authorized for one representative.

        Task 4 must build relation terms only from this allowlist, never from
        the representative's unrestricted mathematical orbit.
        """
        if representative not in self.representatives:
            raise ValueError("representative is not in this factor base")
        closure = set(raw_closure(fixture, self.mode, (representative,)))
        allowed = tuple(point for point in self.raw_points if point in closure)
        if not allowed:
            raise ValueError("representative has no allowed raw points")
        return allowed


def _require_fixture(fixture: CurveFixture) -> None:
    if not validate_fixture(fixture):
        raise ValueError("fixture must be a valid checked curve fixture")


def _require_mode(mode: Mode) -> None:
    if not isinstance(mode, Mode):
        raise TypeError("mode must be a Mode")


def _require_positive_integer(value: int, name: str) -> None:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def _require_seed(seed: int) -> None:
    if type(seed) is not int:
        raise TypeError("seed must be an integer")


def _require_base_point(fixture: CurveFixture, point: Point) -> tuple[int, int]:
    if point is None:
        raise ValueError("the identity cannot be a factor-base point")
    if not fixture.curve.is_on_curve(point):
        raise ValueError("factor-base point is malformed or off the curve")
    if fixture.curve.mul(fixture.n, point) is not None:
        raise ValueError("factor-base point is outside the fixture subgroup")
    return point


def _rank_digest(field_prime: int, seed: int, point: tuple[int, int]) -> bytes:
    width = (field_prime.bit_length() + 7) // 8
    compressed = bytes((2 | (point[1] & 1),)) + point[0].to_bytes(width, "big")
    seed_bytes = str(seed).encode("ascii")
    encoded = (
        b"owic-factor-base-rank-v2\x00"
        + len(seed_bytes).to_bytes(4, "big")
        + seed_bytes
        + compressed
    )
    return sha256(encoded).digest()


def point_rank(fixture: CurveFixture, seed: int, point: Point) -> bytes:
    _require_fixture(fixture)
    _require_seed(seed)
    point = _require_base_point(fixture, point)
    return _rank_digest(fixture.p, seed, point)


def ranked_subgroup_points(
    fixture: CurveFixture, seed: int
) -> tuple[tuple[int, int], ...]:
    _require_fixture(fixture)
    _require_seed(seed)
    generator = _require_base_point(fixture, fixture.generator)
    points: list[tuple[int, int]] = []
    point: Point = generator
    while point is not None:
        points.append(_require_base_point(fixture, point))
        if len(points) >= fixture.n:
            raise ArithmeticError("subgroup enumeration did not reach the identity")
        point = fixture.curve.add(point, generator)
    if len(points) != fixture.n - 1 or len(set(points)) != len(points):
        raise ArithmeticError("fixture generator does not enumerate its declared subgroup")
    return tuple(
        sorted(points, key=lambda item: (_rank_digest(fixture.p, seed, item), item))
    )


def negation_representative(
    fixture: CurveFixture, point: Point
) -> tuple[int, int]:
    _require_fixture(fixture)
    point = _require_base_point(fixture, point)
    negative = _require_base_point(fixture, fixture.curve.neg(point))
    return min(point, negative)


def representative_for(
    fixture: CurveFixture, mode: Mode, point: Point
) -> tuple[int, int]:
    _require_fixture(fixture)
    _require_mode(mode)
    point = _require_base_point(fixture, point)
    if mode is Mode.CLASSIC:
        return point
    if mode is Mode.NEGATION:
        return negation_representative(fixture, point)
    representative = canonicalize(fixture, point).representative
    return _require_base_point(fixture, representative)


def choose_representatives(
    fixture: CurveFixture,
    mode: Mode,
    ranked_points: Iterable[Point],
    count: int,
) -> tuple[tuple[int, int], ...]:
    _require_fixture(fixture)
    _require_mode(mode)
    _require_positive_integer(count, "count")
    selected: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    for point in ranked_points:
        representative = representative_for(fixture, mode, point)
        if representative in seen:
            continue
        seen.add(representative)
        selected.append(representative)
        if len(selected) == count:
            return tuple(selected)
    raise ValueError(
        f"requested {count} representatives exceeds {mode.value} quotient capacity"
    )


def raw_closure(
    fixture: CurveFixture, mode: Mode, representatives: Iterable[Point]
) -> tuple[tuple[int, int], ...]:
    _require_fixture(fixture)
    _require_mode(mode)
    points: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    representative_count = 0
    for representative in representatives:
        representative_count += 1
        representative = _require_base_point(fixture, representative)
        if representative_for(fixture, mode, representative) != representative:
            raise ValueError("representatives must be canonical for their mode")
        if mode is Mode.CLASSIC:
            members = (representative,)
        elif mode is Mode.NEGATION:
            members = (representative, fixture.curve.neg(representative))
        else:
            members = orbit_points(fixture, representative)
        for point in members:
            point = _require_base_point(fixture, point)
            if point not in seen:
                seen.add(point)
                points.append(point)
    if representative_count == 0:
        raise ValueError("representatives must be nonempty")
    return tuple(points)


def build_comparison_bases(
    fixture: CurveFixture,
    budget_type: BudgetType,
    budget: int,
    seed: int,
) -> dict[Mode, FactorBase]:
    _require_fixture(fixture)
    if not isinstance(budget_type, BudgetType):
        raise TypeError("budget_type must be a BudgetType")
    _require_positive_integer(budget, "budget")
    _require_seed(seed)
    ranked_points = ranked_subgroup_points(fixture, seed)

    if budget_type is BudgetType.EQUAL_RAW:
        if budget > len(ranked_points):
            raise ValueError("equal-raw budget exceeds the nonidentity subgroup size")
        shared_raw = ranked_points[:budget]
        bases: dict[Mode, FactorBase] = {}
        for mode in Mode:
            representatives = tuple(
                dict.fromkeys(
                    representative_for(fixture, mode, point) for point in shared_raw
                )
            )
            bases[mode] = FactorBase(
                mode,
                representatives,
                shared_raw,
                len(representatives),
                0,
            )
        return bases

    bases = {}
    for mode in Mode:
        representatives = choose_representatives(
            fixture, mode, ranked_points, budget
        )
        bases[mode] = FactorBase(
            mode,
            representatives,
            raw_closure(fixture, mode, representatives),
            budget,
            budget if budget_type is BudgetType.EQUAL_WORK else 0,
        )
    return bases
