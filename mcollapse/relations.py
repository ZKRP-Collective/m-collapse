"""Exact relation construction with immutable, bounded search preparation."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field, fields
from enum import Enum
from itertools import islice, product
from types import MappingProxyType
from typing import Iterable, Mapping

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
from mcollapse.curves import CurveFixture, Point, validate_fixture
from mcollapse.factor_base import FactorBase, Mode
from mcollapse.orbits import canonicalize, phi_power


def _is_affine_point(point: Point) -> bool:
    return (
        isinstance(point, tuple)
        and len(point) == 2
        and all(type(coordinate) is int for coordinate in point)
    )


@dataclass(frozen=True)
class Term:
    point: Point
    column: int
    coefficient: int
    cost: CostVector

    def __post_init__(self) -> None:
        if not _is_affine_point(self.point):
            raise ValueError("term point must be an affine integer pair")
        if type(self.column) is not int or self.column < 0:
            raise ValueError("term column must be a nonnegative integer")
        if type(self.coefficient) is not int:
            raise TypeError("term coefficient must be an integer")
        if not isinstance(self.cost, CostVector):
            raise TypeError("term cost must be a CostVector")


@dataclass(frozen=True)
class TermCatalog:
    terms: tuple[Term, ...]
    attempted_terms: int
    construction_cost: CostVector

    def __post_init__(self) -> None:
        if not isinstance(self.terms, tuple) or not all(
            isinstance(term, Term) for term in self.terms
        ):
            raise ValueError("terms must be a tuple of Term values")
        if type(self.attempted_terms) is not int or self.attempted_terms < len(self.terms):
            raise ValueError("attempted_terms must cover every unique term")
        if not isinstance(self.construction_cost, CostVector):
            raise TypeError("construction_cost must be a CostVector")

    def __iter__(self):
        return iter(self.terms)

    def __len__(self) -> int:
        return len(self.terms)


@dataclass(frozen=True)
class Relation:
    terms: tuple[Term, ...]
    normalized_row: tuple[int, ...]
    cost: CostVector

    def __post_init__(self) -> None:
        if (
            not isinstance(self.terms, tuple)
            or not self.terms
            or not all(isinstance(term, Term) for term in self.terms)
        ):
            raise ValueError("relation terms must be a nonempty tuple of terms")
        if (
            not isinstance(self.normalized_row, tuple)
            or not self.normalized_row
            or not all(type(value) is int and value >= 0 for value in self.normalized_row)
        ):
            raise ValueError("normalized row must be a nonempty tuple of nonnegative integers")
        if not isinstance(self.cost, CostVector):
            raise TypeError("relation cost must be a CostVector")


class CombinationBranch(str, Enum):
    ORDINARY_ADD = "ordinary-add"
    DOUBLING = "doubling"
    INVERSE_CANCELLATION = "inverse-cancellation"
    IDENTITY_NOOP = "identity-noop"


@dataclass(frozen=True)
class PointCombination:
    point: Point
    branch: CombinationBranch
    cost: CostVector


@dataclass(frozen=True)
class SearchResult:
    relations: tuple[Relation, ...]
    search_cost: CostVector
    search_candidates: int

    def __post_init__(self) -> None:
        if not isinstance(self.relations, tuple) or not all(
            isinstance(relation, Relation) for relation in self.relations
        ):
            raise ValueError("relations must be a tuple of Relation values")
        if not isinstance(self.search_cost, CostVector):
            raise TypeError("search_cost must be a CostVector")
        if type(self.search_candidates) is not int or self.search_candidates < 0:
            raise ValueError("search_candidates must be nonnegative")

    def __iter__(self):
        return iter(self.relations)

    def __len__(self) -> int:
        return len(self.relations)


_IndexValue = Term | tuple[Term, Term]


@dataclass(frozen=True)
class PreparedSearch:
    fixture: CurveFixture
    base: FactorBase
    arity: int
    candidate_limit: int
    catalog: TermCatalog
    column_count: int
    basis_points: tuple[tuple[int, int], ...]
    preparation_cost: CostVector
    preparation_candidates: int
    search_candidate_limit: int
    _index: Mapping[Point, tuple[_IndexValue, ...]] = field(repr=False)
    _term_allowlist: frozenset[Term] = field(repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self._index, MappingProxyType):
            raise TypeError("prepared index must be immutable")
        if len(self.basis_points) != self.column_count:
            raise ValueError("physical basis width must match the matrix width")
        if self._term_allowlist != frozenset(self.catalog.terms):
            raise ValueError("term allowlist must exactly match the prepared catalog")
        if self.preparation_candidates + self.search_candidate_limit > self.candidate_limit:
            raise ValueError("prepared candidate allocation exceeds its limit")

    def search(self, target: Point) -> SearchResult:
        """Search one target without mutating or rebuilding prepared state.

        Candidate units count index/pair insertions during preparation, then
        one unit for every target lookup and one for every relation hit that is
        normalized and verified. No result can exceed ``candidate_limit``.
        """
        target = _require_subgroup_point(self.fixture, target, "target")
        found: dict[tuple[int, ...], Relation] = {}
        search_cost = CostVector()
        search_candidates = 0

        for term in self.catalog.terms:
            if search_candidates >= self.search_candidate_limit:
                break
            needed = _combine_checked(
                self.fixture, target, self.fixture.curve.neg(term.point)
            )
            search_cost += needed.cost + PROBE
            search_candidates += 1
            hits = self._index.get(needed.point, ())
            for hit in hits:
                if search_candidates >= self.search_candidate_limit:
                    break
                search_candidates += 1
                search_cost += PROBE
                if isinstance(hit, Term):
                    if self.arity != 2:
                        raise ArithmeticError("pair index contains a direct term")
                    relation_terms = (term, hit)
                else:
                    if self.arity != 3:
                        raise ArithmeticError("direct index contains a term pair")
                    relation_terms = (hit[0], hit[1], term)
                ordered_terms = tuple(sorted(relation_terms, key=_term_key))
                relation, construction_cost = _relation_from_terms_with_work(
                    self.fixture, ordered_terms, self.column_count
                )
                search_cost += construction_cost
                verified, verification_cost = _verify_relation_with_cost(
                    self, relation, target
                )
                search_cost += verification_cost
                if not verified:
                    raise ArithmeticError("prepared lookup produced an invalid relation")
                previous = found.get(relation.normalized_row)
                if previous is None or _relation_key(relation) < _relation_key(previous):
                    found[relation.normalized_row] = relation

        return SearchResult(
            tuple(found[row] for row in sorted(found)),
            search_cost,
            search_candidates,
        )


def _require_fixture(fixture: CurveFixture) -> None:
    if not validate_fixture(fixture):
        raise ValueError("fixture must be a valid checked curve fixture")


def _require_curve_point(fixture: CurveFixture, point: Point, name: str) -> Point:
    if point is not None and (
        not _is_affine_point(point) or not fixture.curve.is_on_curve(point)
    ):
        raise ValueError(f"{name} must be the identity or an affine curve point")
    return point


def _require_subgroup_point(
    fixture: CurveFixture, point: Point, name: str
) -> tuple[int, int]:
    if not _is_affine_point(point) or not fixture.curve.is_on_curve(point):
        raise ValueError(f"{name} must be a nonidentity point on the curve")
    if fixture.curve.mul(fixture.n, point) is not None:
        raise ValueError(f"{name} must belong to the fixture subgroup")
    return point


def _combine_checked(
    fixture: CurveFixture, left: Point, right: Point
) -> PointCombination:
    if left is None or right is None:
        return PointCombination(
            right if left is None else left,
            CombinationBranch.IDENTITY_NOOP,
            IDENTITY_NOOP,
        )
    if left[0] == right[0] and (left[1] + right[1]) % fixture.p == 0:
        return PointCombination(
            None, CombinationBranch.INVERSE_CANCELLATION, CANCELLATION
        )
    if left == right:
        return PointCombination(
            fixture.curve.add(left, right),
            CombinationBranch.DOUBLING,
            AFFINE_DOUBLE,
        )
    return PointCombination(
        fixture.curve.add(left, right),
        CombinationBranch.ORDINARY_ADD,
        AFFINE_ADD,
    )


def combine_points(
    fixture: CurveFixture, left: Point, right: Point
) -> PointCombination:
    _require_fixture(fixture)
    left = _require_curve_point(fixture, left, "left point")
    right = _require_curve_point(fixture, right, "right point")
    return _combine_checked(fixture, left, right)


def normalize_row(values: tuple[int, ...], modulus: int) -> tuple[int, ...]:
    if not isinstance(values, tuple) or not values:
        raise ValueError("row values must be a nonempty tuple")
    if not all(type(value) is int for value in values):
        raise TypeError("row coefficients must be integers")
    if type(modulus) is not int or modulus <= 1:
        raise ValueError("modulus must be an integer greater than one")
    return tuple(value % modulus for value in values)


def build_terms(fixture: CurveFixture, base: FactorBase) -> TermCatalog:
    _require_fixture(fixture)
    if not isinstance(base, FactorBase):
        raise TypeError("base must be a FactorBase")

    terms: list[Term] = []
    seen: set[tuple[tuple[int, int], int, int]] = set()
    attempted_terms = 0
    construction_cost = CostVector()
    for index, representative in enumerate(base.representatives):
        representative = _require_subgroup_point(fixture, representative, "representative")
        for source in base.allowed_points(fixture, representative):
            source = _require_subgroup_point(fixture, source, "allowed point")
            if base.mode is Mode.CLASSIC:
                attempts = ((source, index, 1, CostVector()),)
            elif base.mode is Mode.NEGATION:
                if source == representative:
                    sign = 1
                elif source == fixture.curve.neg(representative):
                    sign = -1
                else:
                    raise ValueError("negation allowlist point does not match its representative")
                attempts = ((source, index, sign, CostVector()),)
            else:
                encoding = canonicalize(fixture, source)
                if encoding.representative != representative:
                    raise ValueError("orbit allowlist point does not match its representative")
                transform_cost = PHI * encoding.exponent
                if base.mode is Mode.GLV_EXPANDED:
                    attempts = (
                        (
                            source,
                            3 * index + encoding.exponent,
                            encoding.sign,
                            transform_cost,
                        ),
                    )
                else:
                    multiplier = (
                        encoding.sign
                        * pow(fixture.eigenvalue, encoding.exponent, fixture.n)
                    ) % fixture.n
                    weights = (
                        (-2, -1, 1, 2)
                        if base.mode is Mode.MCOLLAPSE_WEIGHTED
                        else (1,)
                    )
                    attempts = tuple(
                        (
                            fixture.curve.mul(weight, source),
                            index,
                            weight * multiplier % fixture.n,
                            transform_cost
                            + (AFFINE_DOUBLE if abs(weight) == 2 else CostVector()),
                        )
                        for weight in weights
                    )

            for point, column, coefficient, cost in attempts:
                attempted_terms += 1
                construction_cost += cost
                if point is None:
                    continue
                key = (point, column, coefficient)
                if key in seen:
                    continue
                seen.add(key)
                terms.append(Term(point, column, coefficient, cost))
    return TermCatalog(tuple(terms), attempted_terms, construction_cost)


def _row_from_terms(
    terms: tuple[Term, ...], column_count: int
) -> tuple[int, ...] | None:
    row = [0] * column_count
    for term in terms:
        if term.column >= column_count:
            return None
        row[term.column] += term.coefficient
    return tuple(row)


def _relation_from_terms_with_work(
    fixture: CurveFixture, terms: tuple[Term, ...], column_count: int
) -> tuple[Relation, CostVector]:
    _require_fixture(fixture)
    if not isinstance(terms, tuple) or not terms:
        raise ValueError("terms must be a nonempty tuple")
    if not all(isinstance(term, Term) for term in terms):
        raise TypeError("terms must contain only Term values")
    if type(column_count) is not int or column_count <= 0:
        raise ValueError("column_count must be a positive integer")

    raw_row = _row_from_terms(terms, column_count)
    if raw_row is None:
        raise ValueError("term column is outside the relation width")
    work = NORMALIZE * column_count
    cost = work
    total: Point = None
    for term in terms:
        _require_subgroup_point(fixture, term.point, "term point")
        cost += term.cost
        combination = _combine_checked(fixture, total, term.point)
        total = combination.point
        cost += combination.cost
        work += combination.cost
    relation = Relation(terms, normalize_row(raw_row, fixture.n), cost)
    return relation, work


def relation_from_terms(
    fixture: CurveFixture, terms: tuple[Term, ...], column_count: int
) -> Relation:
    relation, _ = _relation_from_terms_with_work(fixture, terms, column_count)
    return relation


def _multiply_point_with_cost(
    fixture: CurveFixture, coefficient: int, point: tuple[int, int]
) -> tuple[Point, CostVector]:
    result: Point = None
    addend: Point = point
    cost = CostVector()
    remaining = coefficient
    while remaining:
        if remaining & 1:
            combination = _combine_checked(fixture, result, addend)
            result = combination.point
            cost += combination.cost
        remaining >>= 1
        if remaining:
            combination = _combine_checked(fixture, addend, addend)
            addend = combination.point
            cost += combination.cost
    return result, cost


def _verify_relation_with_cost(
    prepared: PreparedSearch, relation: Relation, target: tuple[int, int]
) -> tuple[bool, CostVector]:
    fixture = prepared.fixture
    column_count = prepared.column_count
    if len(relation.normalized_row) != column_count:
        return False, CostVector()
    if not all(
        type(value) is int and 0 <= value < fixture.n
        for value in relation.normalized_row
    ):
        return False, CostVector()
    if not all(term in prepared._term_allowlist for term in relation.terms):
        return False, CostVector()

    cost = NORMALIZE * column_count
    raw_row = _row_from_terms(relation.terms, column_count)
    if raw_row is None:
        return False, cost
    expected_row = normalize_row(raw_row, fixture.n)
    if expected_row != relation.normalized_row:
        return False, cost

    term_total: Point = None
    for term in relation.terms:
        point = _require_subgroup_point(fixture, term.point, "term point")
        combination = _combine_checked(fixture, term_total, point)
        term_total = combination.point
        cost += combination.cost

    row_total: Point = None
    for coefficient, basis_point in zip(
        relation.normalized_row, prepared.basis_points, strict=True
    ):
        if coefficient == 0:
            continue
        multiple, multiplication_cost = _multiply_point_with_cost(
            fixture, coefficient, basis_point
        )
        cost += multiplication_cost
        combination = _combine_checked(fixture, row_total, multiple)
        row_total = combination.point
        cost += combination.cost
    return term_total == target and row_total == target, cost


def verify_relation(
    prepared: PreparedSearch, relation: Relation, target: Point
) -> bool:
    if not isinstance(prepared, PreparedSearch):
        raise TypeError("prepared must be a PreparedSearch")
    if not isinstance(relation, Relation):
        raise TypeError("relation must be a Relation")
    target = _require_subgroup_point(prepared.fixture, target, "target")
    verified, _ = _verify_relation_with_cost(prepared, relation, target)
    return verified


def _cost_key(cost: CostVector) -> tuple[int, ...]:
    return tuple(getattr(cost, item.name) for item in fields(cost))


def _term_key(term: Term) -> tuple[object, ...]:
    return (term.point, term.column, term.coefficient, _cost_key(term.cost))


def _relation_key(relation: Relation) -> tuple[object, ...]:
    return (_cost_key(relation.cost), tuple(_term_key(term) for term in relation.terms))


def prepare_search(
    fixture: CurveFixture,
    base: FactorBase,
    arity: int,
    candidate_limit: int,
) -> PreparedSearch:
    """Prepare an immutable index under one deterministic candidate ledger.

    Both arities reserve half the ledger for preparation and the remainder for
    target search. Arity two spends preparation units on direct-index inserts;
    arity three spends exactly the available units on the ordered pair prefix.
    Unused preparation capacity is not transferred, keeping target budgets
    independent of factor-base size.
    """
    _require_fixture(fixture)
    if not isinstance(base, FactorBase):
        raise TypeError("base must be a FactorBase")
    if type(arity) is not int or arity not in (2, 3):
        raise ValueError("only arities two and three are registered")
    if type(candidate_limit) is not int or candidate_limit <= 0:
        raise ValueError("candidate_limit must be a positive integer")

    catalog = build_terms(fixture, base)
    column_count = len(base.representatives) * (
        3 if base.mode is Mode.GLV_EXPANDED else 1
    )
    if base.mode is Mode.GLV_EXPANDED:
        basis_points = tuple(
            _require_subgroup_point(
                fixture,
                phi_power(fixture, representative, exponent),
                "physical basis point",
            )
            for representative in base.representatives
            for exponent in range(3)
        )
    else:
        basis_points = tuple(
            _require_subgroup_point(fixture, representative, "physical basis point")
            for representative in base.representatives
        )
    preparation_budget = candidate_limit // 2
    search_candidate_limit = candidate_limit - preparation_budget
    preparation_cost = catalog.construction_cost
    preparation_candidates = 0
    mutable_index: dict[Point, list[_IndexValue]] = defaultdict(list)

    if arity == 2:
        for term in catalog.terms[:preparation_budget]:
            mutable_index[term.point].append(term)
            preparation_cost += PROBE
            preparation_candidates += 1
    else:
        pair_attempts = islice(
            product(catalog.terms, repeat=2), preparation_budget
        )
        for left, right in pair_attempts:
            combination = _combine_checked(fixture, left.point, right.point)
            mutable_index[combination.point].append((left, right))
            preparation_cost += combination.cost + PROBE
            preparation_candidates += 1

    immutable_index = MappingProxyType(
        {point: tuple(values) for point, values in mutable_index.items()}
    )
    return PreparedSearch(
        fixture,
        base,
        arity,
        candidate_limit,
        catalog,
        column_count,
        basis_points,
        preparation_cost,
        preparation_candidates,
        search_candidate_limit,
        immutable_index,
        frozenset(catalog.terms),
    )


def aggregate_cost(
    prepared: PreparedSearch, results: Iterable[SearchResult]
) -> CostVector:
    if not isinstance(prepared, PreparedSearch):
        raise TypeError("prepared must be a PreparedSearch")
    total = prepared.preparation_cost
    for result in results:
        if not isinstance(result, SearchResult):
            raise TypeError("results must contain only SearchResult values")
        total += result.search_cost
    return total


def aggregate_candidates(
    prepared: PreparedSearch, results: Iterable[SearchResult]
) -> int:
    if not isinstance(prepared, PreparedSearch):
        raise TypeError("prepared must be a PreparedSearch")
    total = prepared.preparation_candidates
    for result in results:
        if not isinstance(result, SearchResult):
            raise TypeError("results must contain only SearchResult values")
        total += result.search_candidates
    return total


def find_relations(
    fixture: CurveFixture,
    base: FactorBase,
    target: Point,
    arity: int,
    candidate_limit: int,
) -> tuple[PreparedSearch, SearchResult]:
    prepared = prepare_search(fixture, base, arity, candidate_limit)
    return prepared, prepared.search(target)
