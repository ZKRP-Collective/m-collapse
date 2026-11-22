import unittest
from dataclasses import FrozenInstanceError

from mcollapse.curves import CurveFixture, secp256k1_identity_fixture, tiny_fixture
from mcollapse.orbits import (
    OrbitEncoding,
    canonicalize,
    coefficient_multiplier,
    orbit_points,
    phi,
    phi_power,
)


class EndomorphismTests(unittest.TestCase):
    def test_phi_matches_eigenvalue_and_has_order_three_on_f7_subgroup(self):
        fixture = tiny_fixture()

        for scalar in range(1, fixture.n):
            with self.subTest(scalar=scalar):
                point = fixture.curve.mul(scalar, fixture.generator)
                images = tuple(phi_power(fixture, point, exponent) for exponent in range(4))

                self.assertEqual(images[1], fixture.curve.mul(fixture.eigenvalue, point))
                self.assertEqual(images[3], point)
                self.assertTrue(all(fixture.curve.is_on_curve(image) for image in images))

    def test_phi_power_reduces_all_integer_exponents_modulo_three(self):
        fixture = tiny_fixture()
        point = fixture.generator

        for exponent in range(-7, 8):
            with self.subTest(exponent=exponent):
                self.assertEqual(
                    phi_power(fixture, point, exponent),
                    phi_power(fixture, point, exponent % 3),
                )
        self.assertIsNone(phi(fixture, None))
        self.assertIsNone(phi_power(fixture, None, -1))

    def test_phi_power_rejects_non_integer_exponents(self):
        fixture = tiny_fixture()

        for exponent in (1.0, "1", True):
            with self.subTest(exponent=exponent):
                with self.assertRaisesRegex(TypeError, "^exponent must be an integer$"):
                    phi_power(fixture, fixture.generator, exponent)


class OrbitEnumerationTests(unittest.TestCase):
    def test_orbit_points_have_exact_transformation_order(self):
        fixture = tiny_fixture()

        self.assertEqual(
            orbit_points(fixture, fixture.generator),
            ((1, 2), (1, 5), (2, 2), (2, 5), (4, 2), (4, 5)),
        )

    def test_order_three_point_on_explicit_j0_curve_has_short_deduplicated_orbit(self):
        fixture = CurveFixture(7, 1, 3, (0, 1), 2, 1, 0)

        self.assertTrue(fixture.curve.is_on_curve(fixture.generator))
        self.assertEqual(orbit_points(fixture, fixture.generator), ((0, 1), (0, 6)))

    def test_factor_base_orbits_reject_infinity_and_off_curve_points(self):
        fixture = tiny_fixture()

        for point in (None, (0, 0)):
            with self.subTest(point=point):
                with self.assertRaises(ValueError):
                    orbit_points(fixture, point)


class CanonicalizationTests(unittest.TestCase):
    def test_every_f7_orbit_member_has_one_x_key_and_representative(self):
        fixture = tiny_fixture()

        for scalar in range(1, fixture.n):
            point = fixture.curve.mul(scalar, fixture.generator)
            members = orbit_points(fixture, point)
            encodings = tuple(canonicalize(fixture, member) for member in members)
            expected_representative = min(members, key=lambda member: (member[0], member[1]))

            with self.subTest(scalar=scalar):
                self.assertEqual({encoding.key for encoding in encodings}, {expected_representative[0]})
                self.assertEqual(
                    {encoding.representative for encoding in encodings},
                    {expected_representative},
                )
                self.assertTrue(all(encoding.orbit_size == len(members) for encoding in encodings))
                self.assertEqual(
                    tuple(encoding.reconstruct(fixture) for encoding in encodings),
                    members,
                )
                self.assertTrue(
                    all(
                        coefficient_multiplier(fixture, encoding)
                        == encoding.sign
                        * pow(fixture.eigenvalue, encoding.exponent, fixture.n)
                        % fixture.n
                        for encoding in encodings
                    )
                )

    def test_encodings_reconstruct_exact_inputs_with_distinct_metadata(self):
        fixture = tiny_fixture()
        members = orbit_points(fixture, fixture.generator)
        encodings = tuple(canonicalize(fixture, member) for member in members)

        self.assertEqual(
            tuple(encoding.reconstruct(fixture) for encoding in encodings),
            members,
        )
        self.assertEqual(
            len({(encoding.sign, encoding.exponent) for encoding in encodings}),
            len(members),
        )
        for encoding in encodings:
            self.assertEqual(
                coefficient_multiplier(fixture, encoding),
                encoding.sign * pow(fixture.eigenvalue, encoding.exponent, fixture.n)
                % fixture.n,
            )

    def test_short_orbit_encoding_records_actual_size_and_reconstructs_both_points(self):
        fixture = CurveFixture(7, 1, 3, (0, 1), 2, 1, 0)
        members = orbit_points(fixture, fixture.generator)
        encodings = tuple(canonicalize(fixture, member) for member in members)

        self.assertEqual([encoding.orbit_size for encoding in encodings], [2, 2])
        self.assertEqual(
            tuple(encoding.reconstruct(fixture) for encoding in encodings),
            members,
        )

    def test_encoding_is_frozen(self):
        encoding = canonicalize(tiny_fixture(), tiny_fixture().generator)

        with self.assertRaises(FrozenInstanceError):
            encoding.sign = -1

    def test_canonicalization_rejects_infinity_and_off_curve_points(self):
        fixture = tiny_fixture()

        for point in (None, (0, 0)):
            with self.subTest(point=point):
                with self.assertRaises(ValueError):
                    canonicalize(fixture, point)

    def test_canonicalization_is_deterministic_across_repeated_calls(self):
        fixture = tiny_fixture()
        point = fixture.curve.mul(5, fixture.generator)

        self.assertEqual(
            [canonicalize(fixture, point) for _ in range(3)],
            [canonicalize(fixture, point)] * 3,
        )


class PublicIdentitySpotChecks(unittest.TestCase):
    def test_secp256k1_generator_and_orbit_use_public_identity_only(self):
        fixture = secp256k1_identity_fixture()
        generator = fixture.generator
        members = orbit_points(fixture, generator)
        encodings = tuple(canonicalize(fixture, member) for member in members)

        self.assertEqual(phi(fixture, generator), fixture.curve.mul(fixture.eigenvalue, generator))
        self.assertEqual(phi_power(fixture, generator, 3), generator)
        self.assertEqual(len(members), 6)
        self.assertTrue(all(fixture.curve.is_on_curve(member) for member in members))
        self.assertEqual(len({encoding.key for encoding in encodings}), 1)
        self.assertEqual(
            tuple(encoding.reconstruct(fixture) for encoding in encodings),
            members,
        )


if __name__ == "__main__":
    unittest.main()
