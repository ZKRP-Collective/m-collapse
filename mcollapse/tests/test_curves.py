import unittest

from mcollapse import curves
from mcollapse.curves import (
    Curve,
    CurveFixture,
    curve_order,
    generate_fixture,
    is_prime,
    prime_factors,
    secp256k1_identity_fixture,
    sqrt_mod,
    tiny_fixture,
    validate_fixture,
)


class CurveTests(unittest.TestCase):
    def test_f7_fixture_has_expected_group_and_generator(self):
        fixture = tiny_fixture()

        self.assertEqual(fixture, CurveFixture(7, 3, 13, (1, 2), 2, 3, 0))
        self.assertTrue(validate_fixture(fixture))
        self.assertIsNone(fixture.curve.mul(fixture.n, fixture.generator))

    def test_affine_group_law_on_every_tiny_subgroup_point(self):
        fixture = tiny_fixture()
        points = [fixture.curve.mul(k, fixture.generator) for k in range(fixture.n)]

        self.assertEqual(len(set(points)), 13)
        self.assertIn(None, points)
        for a in range(fixture.n):
            for b in range(fixture.n):
                self.assertEqual(
                    fixture.curve.add(points[a], points[b]),
                    points[(a + b) % fixture.n],
                )

    def test_identity_negation_and_negative_scalars(self):
        fixture = tiny_fixture()
        curve = fixture.curve
        point = fixture.generator

        self.assertEqual(curve.add(None, point), point)
        self.assertEqual(curve.add(point, None), point)
        self.assertIsNone(curve.neg(None))
        self.assertIsNone(curve.add(point, curve.neg(point)))
        self.assertEqual(curve.mul(-3, point), curve.neg(curve.mul(3, point)))
        self.assertIsNone(curve.mul(0, point))

    def test_off_curve_and_malformed_points_are_rejected(self):
        curve = tiny_fixture().curve

        self.assertFalse(curve.is_on_curve((0, 0)))
        self.assertFalse(curve.is_on_curve((1,)))
        self.assertFalse(curve.is_on_curve("not a point"))
        for operation in (
            lambda: curve.neg((0, 0)),
            lambda: curve.add((1, 2), (0, 0)),
            lambda: curve.add((1,), (1, 2)),
            lambda: curve.mul(2, (0, 0)),
        ):
            with self.subTest(operation=operation):
                with self.assertRaises(ValueError):
                    operation()

    def test_non_integer_scalar_is_rejected(self):
        with self.assertRaises(TypeError):
            tiny_fixture().curve.mul(1.5, tiny_fixture().generator)

    def test_doubling_point_with_zero_y_returns_infinity(self):
        curve = Curve(5, 1)

        self.assertTrue(curve.is_on_curve((4, 0)))
        self.assertIsNone(curve.add((4, 0), (4, 0)))
        self.assertIsNone(curve.mul(2, (4, 0)))

    def test_curve_order_counts_identity_and_affine_points(self):
        self.assertEqual(curve_order(Curve(7, 3)), 13)

    def test_project_owned_secp256k1_fixture_checks_identity_only(self):
        fixture = secp256k1_identity_fixture()
        phi_generator = (
            fixture.beta * fixture.generator[0] % fixture.p,
            fixture.generator[1],
        )

        self.assertTrue(validate_fixture(fixture))
        self.assertEqual(phi_generator, fixture.curve.mul(fixture.eigenvalue, fixture.generator))


class NumberTheoryTests(unittest.TestCase):
    def test_primality_is_deterministic(self):
        values = {
            -1: False,
            0: False,
            1: False,
            2: True,
            3: True,
            4: False,
            97: True,
            221: False,
        }

        self.assertEqual({value: is_prime(value) for value in values}, values)
        self.assertEqual([is_prime(104729) for _ in range(2)], [True, True])

    def test_large_strong_pseudoprime_is_composite(self):
        self.assertFalse(is_prime(318665857834031151167461))

    def test_large_perfect_square_is_composite(self):
        self.assertFalse(is_prime(((1 << 64) + 1) ** 2))

    def test_primality_boundary_at_64_bits(self):
        self.assertTrue(is_prime((1 << 64) - 59))
        self.assertFalse(is_prime((1 << 64) + 1))

    def test_prime_factorization_is_deterministic(self):
        self.assertEqual(prime_factors(1), {})
        self.assertEqual(prime_factors(2**4 * 3**2 * 11), {2: 4, 3: 2, 11: 1})
        self.assertEqual(prime_factors(1009), {1009: 1})
        self.assertEqual(prime_factors(1009), prime_factors(1009))

    def test_tonelli_shanks_returns_deterministic_roots(self):
        self.assertEqual(sqrt_mod(0, 13), 0)
        self.assertEqual(sqrt_mod(10, 13), 7)
        self.assertEqual(sqrt_mod(10, 13), sqrt_mod(10, 13))
        self.assertIsNone(sqrt_mod(2, 13))
        self.assertEqual(sqrt_mod(4, 7), 2)

    def test_square_root_requires_an_odd_prime_modulus(self):
        for modulus in (2, 9):
            with self.subTest(modulus=modulus):
                with self.assertRaisesRegex(ValueError, "^modulus must be an odd prime$"):
                    sqrt_mod(4, modulus)


class FixtureGenerationTests(unittest.TestCase):
    def test_subgroup_bit_bounds_are_accepted_by_input_validation(self):
        self.assertIsNone(curves._validate_subgroup_bits(2))
        self.assertIsNone(curves._validate_subgroup_bits(16))

    def test_fixture_generation_rejects_unsupported_subgroup_sizes(self):
        message = "subgroup_bits must be between 2 and 16 inclusive"
        for subgroup_bits in (1, 17):
            with self.subTest(subgroup_bits=subgroup_bits):
                with self.assertRaises(ValueError) as error:
                    generate_fixture(subgroup_bits, 20260804)
                self.assertEqual(str(error.exception), message)

    def test_generated_fixture_is_repeatable_and_has_requested_size(self):
        first = generate_fixture(10, 20260804)
        second = generate_fixture(10, 20260804)

        self.assertEqual(first, second)
        self.assertIn(first.n.bit_length(), {9, 10, 11})
        self.assertTrue(validate_fixture(first))
        self.assertEqual(prime_factors(curve_order(first.curve))[first.n], 1)


if __name__ == "__main__":
    unittest.main()
