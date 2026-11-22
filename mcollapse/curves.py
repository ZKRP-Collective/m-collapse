from __future__ import annotations

from dataclasses import dataclass
from functools import cache
from math import gcd, isqrt
from random import Random

Point = tuple[int, int] | None


@dataclass(frozen=True)
class Curve:
    p: int
    b: int

    def is_on_curve(self, point: Point) -> bool:
        if point is None:
            return True
        if (
            not isinstance(point, tuple)
            or len(point) != 2
            or not all(isinstance(coordinate, int) for coordinate in point)
        ):
            return False
        x, y = point
        return 0 <= x < self.p and 0 <= y < self.p and (y * y - x**3 - self.b) % self.p == 0

    def _require_point(self, point: Point) -> Point:
        if not self.is_on_curve(point):
            raise ValueError("point is malformed or not on the curve")
        return point

    def neg(self, point: Point) -> Point:
        point = self._require_point(point)
        return None if point is None else (point[0], (-point[1]) % self.p)

    def add(self, left: Point, right: Point) -> Point:
        left = self._require_point(left)
        right = self._require_point(right)
        if left is None:
            return right
        if right is None:
            return left

        x1, y1 = left
        x2, y2 = right
        if x1 == x2 and (y1 + y2) % self.p == 0:
            return None
        if left == right:
            if y1 == 0:
                return None
            slope = 3 * x1 * x1 * pow(2 * y1, -1, self.p)
        else:
            slope = (y2 - y1) * pow(x2 - x1, -1, self.p)
        slope %= self.p
        x3 = (slope * slope - x1 - x2) % self.p
        result = (x3, (slope * (x1 - x3) - y1) % self.p)
        if not self.is_on_curve(result):
            raise ArithmeticError("affine addition left the curve")
        return result

    def mul(self, scalar: int, point: Point) -> Point:
        if not isinstance(scalar, int):
            raise TypeError("scalar must be an integer")
        point = self._require_point(point)
        if scalar < 0:
            return self.mul(-scalar, self.neg(point))

        result = None
        addend = point
        while scalar:
            if scalar & 1:
                result = self.add(result, addend)
            addend = self.add(addend, addend)
            scalar >>= 1
        return result


@dataclass(frozen=True)
class CurveFixture:
    p: int
    b: int
    n: int
    generator: Point
    beta: int
    eigenvalue: int
    seed: int

    @property
    def curve(self) -> Curve:
        return Curve(self.p, self.b)


def tiny_fixture() -> CurveFixture:
    return CurveFixture(7, 3, 13, (1, 2), 2, 3, 0)


def secp256k1_identity_fixture() -> CurveFixture:
    return CurveFixture(
        p=0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F,
        b=7,
        n=0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141,
        generator=(
            0x79BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798,
            0x483ADA7726A3C4655DA4FBFC0E1108A8FD17B448A68554199C47D08FFB10D4B8,
        ),
        beta=0x7AE96A2B657C07106E64479EAC3434E99CF0497512F58995C1396C28719501EE,
        eigenvalue=0x5363AD4CC05C30E0A5261C028812645A122E22EA20816678DF02967C1B23BD72,
        seed=0,
    )


def _strong_miller_rabin(value: int, base: int) -> bool:
    odd_part = value - 1
    power = 0
    while odd_part % 2 == 0:
        odd_part //= 2
        power += 1
    residue = pow(base % value, odd_part, value)
    if residue in (1, value - 1):
        return True
    for _ in range(power - 1):
        residue = residue * residue % value
        if residue == value - 1:
            return True
    return False


def _jacobi(symbol: int, modulus: int) -> int:
    symbol %= modulus
    result = 1
    while symbol:
        while symbol % 2 == 0:
            symbol //= 2
            if modulus % 8 in (3, 5):
                result = -result
        symbol, modulus = modulus, symbol
        if symbol % 4 == modulus % 4 == 3:
            result = -result
        symbol %= modulus
    return result if modulus == 1 else 0


def _strong_lucas_prp(value: int) -> bool:
    discriminant = 5
    while True:
        common_factor = gcd(abs(discriminant), value)
        if 1 < common_factor < value:
            return False
        if _jacobi(discriminant, value) == -1:
            break
        discriminant = -(discriminant + 2) if discriminant > 0 else -discriminant + 2

    parameter = (1 - discriminant) // 4
    if gcd(abs(parameter), value) > 1:
        return False
    odd_part = value + 1
    power = 0
    while odd_part % 2 == 0:
        odd_part //= 2
        power += 1

    lucas_u, lucas_v, parameter_power = 0, 2, 1
    for bit in bin(odd_part)[2:]:
        lucas_u = lucas_u * lucas_v % value
        lucas_v = (lucas_v * lucas_v - 2 * parameter_power) % value
        parameter_power = parameter_power * parameter_power % value
        if bit == "1":
            next_u = (lucas_u + lucas_v) % value
            next_v = (discriminant * lucas_u + lucas_v) % value
            lucas_u = (next_u + (value if next_u % 2 else 0)) // 2
            lucas_v = (next_v + (value if next_v % 2 else 0)) // 2
            parameter_power = parameter_power * parameter % value

    if lucas_u == 0 or lucas_v == 0:
        return True
    for _ in range(1, power):
        lucas_v = (lucas_v * lucas_v - 2 * parameter_power) % value
        parameter_power = parameter_power * parameter_power % value
        if lucas_v == 0:
            return True
    return False


@cache
def is_prime(value: int) -> bool:
    """Test primality exactly through 64 bits, then use Baillie-PSW.

    Above 64 bits, a True result means that the value is a Baillie-PSW
    probable prime, not that its primality has been deterministically proved.
    """
    if not isinstance(value, int) or value < 2:
        return False
    small_primes = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37)
    if value in small_primes:
        return True
    if any(value % prime == 0 for prime in small_primes):
        return False
    if value < 1 << 20:
        return all(value % divisor for divisor in range(41, isqrt(value) + 1, 2))
    if isqrt(value) ** 2 == value:
        return False
    if value < 1 << 64:
        return all(
            _strong_miller_rabin(value, base)
            for base in (2, 325, 9375, 28178, 450775, 9780504, 1795265022)
        )
    return _strong_miller_rabin(value, 2) and _strong_lucas_prp(value)


def prime_factors(value: int) -> dict[int, int]:
    if not isinstance(value, int) or value < 1:
        raise ValueError("factorization requires a positive integer")
    factors: dict[int, int] = {}
    divisor = 2
    while divisor * divisor <= value:
        while value % divisor == 0:
            factors[divisor] = factors.get(divisor, 0) + 1
            value //= divisor
        divisor = 3 if divisor == 2 else divisor + 2
    if value > 1:
        factors[value] = factors.get(value, 0) + 1
    return factors


def sqrt_mod(value: int, prime: int) -> int | None:
    if prime <= 2 or not is_prime(prime):
        raise ValueError("modulus must be an odd prime")
    value %= prime
    if value == 0:
        return 0
    if pow(value, (prime - 1) // 2, prime) != 1:
        return None
    if prime % 4 == 3:
        return pow(value, (prime + 1) // 4, prime)

    odd_part = prime - 1
    power = 0
    while odd_part % 2 == 0:
        odd_part //= 2
        power += 1
    non_residue = next(
        candidate
        for candidate in range(2, prime)
        if pow(candidate, (prime - 1) // 2, prime) == prime - 1
    )
    coefficient = pow(non_residue, odd_part, prime)
    root = pow(value, (odd_part + 1) // 2, prime)
    residue = pow(value, odd_part, prime)
    width = power
    while residue != 1:
        index = 1
        square = residue * residue % prime
        while square != 1:
            square = square * square % prime
            index += 1
        factor = pow(coefficient, 1 << (width - index - 1), prime)
        root = root * factor % prime
        residue = residue * factor * factor % prime
        coefficient = factor * factor % prime
        width = index
    return root


def curve_order(curve: Curve) -> int:
    total = 1
    for x in range(curve.p):
        rhs = (x**3 + curve.b) % curve.p
        if rhs == 0:
            total += 1
        elif pow(rhs, (curve.p - 1) // 2, curve.p) == 1:
            total += 2
    return total


def validate_fixture(fixture: CurveFixture) -> bool:
    try:
        if (
            not isinstance(fixture, CurveFixture)
            or not is_prime(fixture.p)
            or fixture.p % 3 != 1
            or not 0 < fixture.b < fixture.p
            or not is_prime(fixture.n)
            or fixture.generator is None
            or not 1 < fixture.beta < fixture.p
            or not 1 < fixture.eigenvalue < fixture.n
        ):
            return False
        curve = fixture.curve
        point = fixture.generator
        phi_point = (fixture.beta * point[0] % fixture.p, point[1])
        return all(
            (
                pow(fixture.beta, 3, fixture.p) == 1,
                curve.is_on_curve(point),
                curve.mul(fixture.n, point) is None,
                curve.mul(fixture.eigenvalue, point) == phi_point,
                (fixture.eigenvalue**2 + fixture.eigenvalue + 1) % fixture.n == 0,
            )
        )
    except (ArithmeticError, TypeError, ValueError):
        return False


def _validate_subgroup_bits(subgroup_bits: int) -> None:
    if not isinstance(subgroup_bits, int) or not 2 <= subgroup_bits <= 16:
        raise ValueError("subgroup_bits must be between 2 and 16 inclusive")


def generate_fixture(subgroup_bits: int, seed: int) -> CurveFixture:
    _validate_subgroup_bits(subgroup_bits)
    rng = Random(seed)
    candidate = (1 << (subgroup_bits + 1)) + 2 * rng.randrange(1 << subgroup_bits) + 1
    for _ in range(20_000):
        while candidate % 3 != 1 or not is_prime(candidate):
            candidate += 2
        prime = candidate
        beta = next(
            (
                root
                for base in range(2, prime)
                if (root := pow(base, (prime - 1) // 3, prime)) not in (0, 1)
            ),
            None,
        )
        if beta is not None:
            for b in range(1, min(prime, 65)):
                curve = Curve(prime, b)
                order = curve_order(curve)
                subgroup_orders = (
                    factor
                    for factor, exponent in prime_factors(order).items()
                    if exponent == 1
                    and factor > 3
                    and abs(factor.bit_length() - subgroup_bits) <= 1
                )
                for subgroup_order in sorted(subgroup_orders, reverse=True):
                    cofactor = order // subgroup_order
                    eigenvalues = (
                        value
                        for value in range(2, subgroup_order)
                        if (value * value + value + 1) % subgroup_order == 0
                    )
                    for x in range(prime):
                        y = sqrt_mod((x**3 + b) % prime, prime)
                        if y is None:
                            continue
                        generator = curve.mul(cofactor, (x, y))
                        if generator is None or curve.mul(subgroup_order, generator) is not None:
                            continue
                        for eigenvalue in eigenvalues:
                            fixture = CurveFixture(
                                prime,
                                b,
                                subgroup_order,
                                generator,
                                beta,
                                eigenvalue,
                                seed,
                            )
                            if validate_fixture(fixture):
                                return fixture
                        break
        candidate += 2
    raise RuntimeError(f"no valid {subgroup_bits}-bit fixture for seed {seed}")
