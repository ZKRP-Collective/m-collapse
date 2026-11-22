from __future__ import annotations

from dataclasses import dataclass

from mcollapse.curves import CurveFixture, Point


def phi(fixture: CurveFixture, point: Point) -> Point:
    return None if point is None else (fixture.beta * point[0] % fixture.p, point[1])


def phi_power(fixture: CurveFixture, point: Point, exponent: int) -> Point:
    if type(exponent) is not int:
        raise TypeError("exponent must be an integer")
    for _ in range(exponent % 3):
        point = phi(fixture, point)
    return point


def orbit_points(fixture: CurveFixture, point: Point) -> tuple[Point, ...]:
    if point is None:
        raise ValueError("the identity has no factor-base orbit")
    if not fixture.curve.is_on_curve(point):
        raise ValueError("point is malformed or not on the curve")

    values: list[Point] = []
    for exponent in range(3):
        transformed = phi_power(fixture, point, exponent)
        values.extend((transformed, fixture.curve.neg(transformed)))
    return tuple(dict.fromkeys(values))


@dataclass(frozen=True)
class OrbitEncoding:
    key: int
    representative: Point
    sign: int
    exponent: int
    orbit_size: int

    def reconstruct(self, fixture: CurveFixture) -> Point:
        point = phi_power(fixture, self.representative, self.exponent)
        return point if self.sign == 1 else fixture.curve.neg(point)


def canonicalize(fixture: CurveFixture, point: Point) -> OrbitEncoding:
    candidates = orbit_points(fixture, point)
    representative = min(candidates, key=lambda item: (item[0], item[1]))
    for exponent in range(3):
        transformed = phi_power(fixture, representative, exponent)
        if transformed == point:
            return OrbitEncoding(representative[0], representative, 1, exponent, len(candidates))
        if fixture.curve.neg(transformed) == point:
            return OrbitEncoding(representative[0], representative, -1, exponent, len(candidates))
    raise ArithmeticError("orbit reconstruction metadata not found")


def coefficient_multiplier(fixture: CurveFixture, encoding: OrbitEncoding) -> int:
    return encoding.sign * pow(fixture.eigenvalue, encoding.exponent, fixture.n) % fixture.n
