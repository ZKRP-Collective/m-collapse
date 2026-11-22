load("mcollapse_sage/rank_guided_mcollapse.sage")
load("mcollapse_sage/quotient_y_summation.sage")


def test_symbolic_line_parameterization_annihilates_polynomial():
    R = PolynomialRing(QQ, names=("b", "s", "t"))
    b, s, t = R.gens()
    A = s^3 + 3 * t
    B = 3 * t^2
    C = t^3 - b * s^3
    assert 27 * (C + b * A)^2 - B * (B + 9 * b)^2 == 0


def test_every_group_relation_is_detected_exactly():
    for subgroup_bits, seed_value in ((7, 53), (8, 59), (9, 61)):
        fixture = find_cm_fixture(subgroup_bits, seed_value)
        _Q, targets = make_public_targets(fixture, 2, seed_value + 1201)
        audit = audit_fixed_target_y_relation(fixture, targets[0].point)
        assert audit["polynomial_zeros"] > 0
        assert audit["exact_phase_relations"] > 0
        assert audit["false_positives"] == 0
        assert audit["false_negatives"] == 0


test_symbolic_line_parameterization_annihilates_polynomial()
test_every_group_relation_is_detected_exactly()
print("quotient_y_summation Sage tests: PASS")
