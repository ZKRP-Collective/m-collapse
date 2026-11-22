load("mcollapse_sage/rank_guided_mcollapse.sage")
load("mcollapse_sage/symmetry_quotient_mcollapse.sage")
load("mcollapse_sage/higher_arity_quotient.sage")


def dense_symbolic_row(row, anchor_count):
    dense = [(ZZ(0), ZZ(0)) for _ in range(anchor_count)]
    for anchor, a, b in row:
        dense[anchor] = (a, b)
    return tuple(dense)


def brute_symbolic_orbits(partial_arity, anchor_count):
    quotient = HigherArityCoefficientQuotient(partial_arity)
    zero = tuple((ZZ(0), ZZ(0)) for _ in range(anchor_count))
    states = {zero}
    for _depth in range(partial_arity):
        next_states = set()
        for row in states:
            for anchor in range(anchor_count):
                for coefficient in quotient.alphabet:
                    updated = list(row)
                    updated[anchor] = eisenstein_add(
                        updated[anchor], coefficient
                    )
                    canonical, _action = canonical_local_tuple(tuple(updated))
                    next_states.add(canonical)
        states = next_states
    return states


def audit_local_witnesses(quotient):
    zero = (ZZ(0), ZZ(0))
    for support, classes in quotient.local_classes.items():
        for values, coordinate_witnesses in classes.items():
            assert sum(len(items) for items in coordinate_witnesses) == quotient.partial_arity
            if support == 0:
                assert values == ()
                total = zero
                for term in coordinate_witnesses[0]:
                    assert term in quotient.alphabet
                    total = eisenstein_add(total, term)
                assert total == zero
                continue
            assert len(values) == support
            assert len(coordinate_witnesses) == support
            for value, witness in zip(values, coordinate_witnesses):
                total = zero
                for term in witness:
                    assert term in quotient.alphabet
                    total = eisenstein_add(total, term)
                assert total == value


def test_closed_higher_arity_counts_and_witnesses():
    arity_four = HigherArityCoefficientQuotient(3)
    arity_five = HigherArityCoefficientQuotient(4)
    assert arity_four.local_class_counts() == {0: 1, 1: 18, 2: 168, 3: 288}
    assert arity_five.local_class_counts() == {
        0: 1,
        1: 32,
        2: 624,
        3: 2880,
        4: 3456,
    }
    assert arity_four.global_count(12) == 74665
    assert arity_five.global_count(12) == 2385889
    audit_local_witnesses(arity_four)
    audit_local_witnesses(arity_five)


def test_direct_generator_matches_brute_symbolic_dp():
    for partial_arity in (3, 4):
        quotient = HigherArityCoefficientQuotient(partial_arity)
        generated = {
            dense_symbolic_row(row, 4)
            for row in quotient.iter_global_rows(4)
        }
        brute = brute_symbolic_orbits(partial_arity, 4)
        assert generated == brute
        assert len(generated) == quotient.global_count(4)


def test_symbolic_catalog_embeds_without_toy_field_collisions():
    fixture = find_cm_fixture(9, 37)
    quotient = HigherArityCoefficientQuotient(4)
    mapped = {
        map_symbolic_row_to_field(row, fixture)
        for row in quotient.iter_global_rows(4)
    }
    assert len(mapped) == quotient.global_count(4)


test_closed_higher_arity_counts_and_witnesses()
test_direct_generator_matches_brute_symbolic_dp()
test_symbolic_catalog_embeds_without_toy_field_collisions()
print("higher_arity_quotient Sage tests: PASS")
