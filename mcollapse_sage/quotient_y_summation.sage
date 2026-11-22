"""GLV-quotient y-coordinate summation relation for j=0 curves."""


def quotient_y_s3(y1, y2, y3, curve_b):
    """Return the symmetric three-point quotient polynomial value.

    For E: y^2 = x^3 + b, a zero is necessary for three GLV phases of
    points with ordinates y1, y2, y3 to sum to the identity.
    """
    A = y1 + y2 + y3
    B = y1 * y2 + y1 * y3 + y2 * y3
    C = y1 * y2 * y3
    return 27 * (C + curve_b * A)^2 - B * (B + 9 * curve_b)^2


def phase_zero_sum_witness(fixture, points):
    """Return relative mu_3 phases whose point sum is zero, if any."""
    if len(points) != 3:
        raise ValueError("the quotient y relation has arity three")
    first = points[0]
    for second_exponent in range(3):
        second = fixture.phi(points[1], second_exponent)
        for third_exponent in range(3):
            third = fixture.phi(points[2], third_exponent)
            if first + second + third == fixture.E(0):
                return (0, second_exponent, third_exponent)
    return None


def subgroup_y_orbit_representatives(fixture):
    """Return one point for each nonzero subgroup mu_3 orbit, keyed by y."""
    fibers = {}
    for scalar in range(1, fixture.n):
        point = scalar * fixture.G
        key = int(point[1])
        fibers.setdefault(key, []).append(point)
    representatives = tuple(
        min(fibers[key], key=point_key) for key in sorted(fibers)
    )
    for point in representatives:
        orbit = {point_key(fixture.phi(point, exponent)) for exponent in range(3)}
        same_y = {point_key(candidate) for candidate in fibers[int(point[1])]}
        if orbit != same_y:
            raise ArithmeticError("subgroup y fiber is not one exact mu_3 orbit")
    return representatives


def audit_fixed_target_y_relation(fixture, target):
    """Exhaustively compare quotient_y_s3 with phase enumeration."""
    representatives = subgroup_y_orbit_representatives(fixture)
    target_inverse = -target
    polynomial_zeros = ZZ(0)
    exact_phase_relations = ZZ(0)
    false_positives = ZZ(0)
    false_negatives = ZZ(0)
    witnesses = []
    for left_index, left in enumerate(representatives):
        for right in representatives[left_index:]:
            value = quotient_y_s3(
                left[1], right[1], target_inverse[1], fixture.Fp(fixture.b)
            )
            witness = phase_zero_sum_witness(
                fixture, (left, right, target_inverse)
            )
            polynomial_zero = value == 0
            exact = witness is not None
            polynomial_zeros += polynomial_zero
            exact_phase_relations += exact
            false_positives += polynomial_zero and not exact
            false_negatives += exact and not polynomial_zero
            if exact:
                witnesses.append(
                    (
                        int(left[1]),
                        int(right[1]),
                        tuple(int(exponent) for exponent in witness),
                    )
                )
    return {
        "y_orbit_representatives": int(len(representatives)),
        "unordered_pairs_checked": int(
            binomial(len(representatives) + 1, 2)
        ),
        "polynomial_zeros": int(polynomial_zeros),
        "exact_phase_relations": int(exact_phase_relations),
        "false_positives": int(false_positives),
        "false_negatives": int(false_negatives),
        "witnesses": tuple(witnesses),
    }
