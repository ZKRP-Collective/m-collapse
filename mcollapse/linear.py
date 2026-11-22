"""Deterministic modular linear algebra and explicit GLV elimination."""

from __future__ import annotations

from collections.abc import Iterable

from mcollapse.curves import is_prime


Row = tuple[int, ...]


def _require_modulus(modulus: int) -> None:
    if type(modulus) is not int:
        raise TypeError("modulus must be an integer")
    if not is_prime(modulus):
        raise ValueError("modulus must be prime")


def _require_eigenvalue(eigenvalue: int) -> None:
    if type(eigenvalue) is not int:
        raise TypeError("eigenvalue must be an integer")


def _require_orbit_count(orbit_count: int) -> None:
    if type(orbit_count) is not int:
        raise TypeError("orbit_count must be an integer")
    if orbit_count <= 0:
        raise ValueError("orbit_count must be positive")


def _validated_rows(rows: Iterable[Iterable[int]]) -> tuple[list[Row], int | None]:
    if isinstance(rows, (str, bytes)):
        raise TypeError("rows must be an iterable of rows")
    try:
        candidates = list(rows)
    except TypeError as error:
        raise TypeError("rows must be an iterable of rows") from error

    validated: list[Row] = []
    width: int | None = None
    for candidate in candidates:
        if isinstance(candidate, (str, bytes)):
            raise TypeError("each row must be an iterable of integers")
        try:
            row = tuple(candidate)
        except TypeError as error:
            raise TypeError("each row must be an iterable of integers") from error
        if not all(type(value) is int for value in row):
            raise TypeError("row cells must be integers")
        if width is None:
            width = len(row)
        elif len(row) != width:
            raise ValueError("rows must have equal widths")
        validated.append(row)
    return validated, width


def rref_mod(rows: Iterable[Iterable[int]], modulus: int) -> list[Row]:
    """Return the unique nonzero reduced row-echelon basis over a prime field."""
    _require_modulus(modulus)
    validated, width = _validated_rows(rows)
    if not validated:
        return []

    matrix = [[value % modulus for value in row] for row in validated]
    pivot_row = 0
    for column in range(width or 0):
        pivot = next(
            (
                index
                for index in range(pivot_row, len(matrix))
                if matrix[index][column]
            ),
            None,
        )
        if pivot is None:
            continue
        matrix[pivot_row], matrix[pivot] = matrix[pivot], matrix[pivot_row]
        inverse = pow(matrix[pivot_row][column], -1, modulus)
        matrix[pivot_row] = [
            value * inverse % modulus for value in matrix[pivot_row]
        ]
        for index, row in enumerate(matrix):
            if index == pivot_row or row[column] == 0:
                continue
            factor = row[column]
            matrix[index] = [
                (value - factor * pivot_value) % modulus
                for value, pivot_value in zip(
                    row, matrix[pivot_row], strict=True
                )
            ]
        pivot_row += 1
        if pivot_row == len(matrix):
            break
    return [tuple(row) for row in matrix if any(row)]


def rank_mod(rows: Iterable[Iterable[int]], modulus: int) -> int:
    return len(rref_mod(rows, modulus))


def canonical_row_space(
    rows: Iterable[Iterable[int]], modulus: int
) -> tuple[Row, ...]:
    return tuple(rref_mod(rows, modulus))


def incremental_ranks(
    rows: Iterable[Iterable[int]], modulus: int
) -> tuple[int, ...]:
    """Return the rank after each input row is incorporated."""
    _require_modulus(modulus)
    validated, width = _validated_rows(rows)
    basis: dict[int, list[int]] = {}
    ranks: list[int] = []
    for row in validated:
        reduced = [value % modulus for value in row]
        for pivot in sorted(basis):
            factor = reduced[pivot]
            if factor:
                reduced = [
                    (value - factor * basis_value) % modulus
                    for value, basis_value in zip(
                        reduced, basis[pivot], strict=True
                    )
                ]
        pivot = next(
            (column for column in range(width or 0) if reduced[column]), None
        )
        if pivot is not None:
            inverse = pow(reduced[pivot], -1, modulus)
            basis[pivot] = [value * inverse % modulus for value in reduced]
        ranks.append(len(basis))
    return tuple(ranks)


def glv_constraint_rows(
    orbit_count: int,
    eigenvalue: int,
    modulus: int,
    include_rhs: bool = True,
) -> list[Row]:
    """Build ell_e - lambda**e ell_0 = 0 for each physical orbit."""
    _require_modulus(modulus)
    _require_eigenvalue(eigenvalue)
    _require_orbit_count(orbit_count)
    if type(include_rhs) is not bool:
        raise TypeError("include_rhs must be a boolean")

    physical_width = 3 * orbit_count
    width = physical_width + int(include_rhs)
    constraints: list[Row] = []
    for orbit in range(orbit_count):
        offset = 3 * orbit
        for exponent in (1, 2):
            row = [0] * width
            row[offset] = -pow(eigenvalue, exponent, modulus) % modulus
            row[offset + exponent] = 1
            constraints.append(tuple(row))
    return constraints


def _eliminate_glv(
    rows: Iterable[Iterable[int]],
    eigenvalue: int,
    modulus: int,
    *,
    has_rhs: bool,
) -> list[Row]:
    _require_modulus(modulus)
    _require_eigenvalue(eigenvalue)
    validated, width = _validated_rows(rows)
    if not validated:
        return []
    if width is None:
        raise ArithmeticError("validated rows lost their width")

    physical_width = width - 1 if has_rhs else width
    if physical_width < 0 or physical_width % 3:
        suffix = " plus one RHS" if has_rhs else ""
        raise ValueError(
            f"expanded GLV rows require three columns per orbit{suffix}"
        )

    powers = (1, eigenvalue % modulus, pow(eigenvalue, 2, modulus))
    orbit_count = physical_width // 3
    eliminated: list[Row] = []
    for row in validated:
        compact = tuple(
            sum(
                row[3 * orbit + exponent] * powers[exponent]
                for exponent in range(3)
            )
            % modulus
            for orbit in range(orbit_count)
        )
        if has_rhs:
            compact += (row[-1] % modulus,)
        eliminated.append(compact)
    return eliminated


def eliminate_glv_columns_with_rhs(
    rows: Iterable[Iterable[int]], eigenvalue: int, modulus: int
) -> list[Row]:
    """Substitute GLV physical columns while preserving the explicit RHS."""
    return _eliminate_glv(
        rows, eigenvalue, modulus, has_rhs=True
    )


def eliminate_glv_coefficients(
    rows: Iterable[Iterable[int]], eigenvalue: int, modulus: int
) -> list[Row]:
    """Substitute GLV physical columns in coefficient-only rows."""
    return _eliminate_glv(
        rows, eigenvalue, modulus, has_rhs=False
    )


def assert_glv_mcollapse_equivalent(
    expanded_rows: Iterable[Iterable[int]],
    compact_rows: Iterable[Iterable[int]],
    eigenvalue: int,
    modulus: int,
    has_rhs: bool,
    orbit_count: int,
) -> None:
    """Raise when expanded GLV and compact MCOLLAPSE row spaces differ."""
    if type(has_rhs) is not bool:
        raise TypeError("has_rhs must be a boolean")
    _require_modulus(modulus)
    _require_eigenvalue(eigenvalue)
    _require_orbit_count(orbit_count)
    expanded, expanded_width = _validated_rows(expanded_rows)
    compact, compact_width = _validated_rows(compact_rows)
    rhs_width = int(has_rhs)
    if (
        expanded_width is not None
        and expanded_width != 3 * orbit_count + rhs_width
    ):
        raise ValueError("expanded GLV row width does not match orbit_count")
    if compact_width is not None and compact_width != orbit_count + rhs_width:
        raise ValueError("compact MCOLLAPSE row width does not match orbit_count")

    eliminated = (
        eliminate_glv_columns_with_rhs(expanded, eigenvalue, modulus)
        if has_rhs
        else eliminate_glv_coefficients(expanded, eigenvalue, modulus)
    )
    if canonical_row_space(eliminated, modulus) != canonical_row_space(
        compact, modulus
    ):
        raise ArithmeticError("GLV/MCOLLAPSE row-space mismatch")
