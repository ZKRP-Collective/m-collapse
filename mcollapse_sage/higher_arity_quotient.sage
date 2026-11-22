"""Exact higher-arity weighted-MCOLLAPSE quotients in the Eisenstein lattice.

The relation lambda^2 + lambda + 1 = 0 is imposed symbolically over ZZ, so
the catalog is independent of a toy finite field.  Permutations disappear by
coefficient-row canonicalization and the diagonal mu_6 action is quotiented
exactly.  This file is SageMath code.
"""


def eisenstein_add(left, right):
    return (ZZ(left[0] + right[0]), ZZ(left[1] + right[1]))


def eisenstein_neg(value):
    return (-ZZ(value[0]), -ZZ(value[1]))


def eisenstein_mul_lambda(value):
    # lambda * (a + b*lambda) = -b + (a-b)*lambda.
    a, b = map(ZZ, value)
    return (-b, a - b)


def eisenstein_unit_actions(value):
    actions = []
    transformed = tuple(map(ZZ, value))
    for exponent in range(3):
        actions.append(transformed)
        actions.append(eisenstein_neg(transformed))
        transformed = eisenstein_mul_lambda(transformed)
    return tuple(actions)


def eisenstein_scale_by_action(value, action_index):
    return eisenstein_unit_actions(value)[ZZ(action_index)]


def canonical_local_tuple(values):
    images = []
    for action_index in range(6):
        image = tuple(
            eisenstein_scale_by_action(value, action_index) for value in values
        )
        images.append((image, ZZ(action_index)))
    return min(images, key=lambda item: (item[0], item[1]))


def positive_compositions(total, parts):
    total = ZZ(total)
    parts = ZZ(parts)
    if parts == 0:
        return ((),) if total == 0 else ()
    if parts == 1:
        return ((total,),) if total >= 1 else ()
    result = []
    for first in range(1, total - parts + 2):
        for tail in positive_compositions(total - first, parts - 1):
            result.append((ZZ(first),) + tuple(tail))
    return tuple(result)


class HigherArityCoefficientQuotient:
    """Generate exact coefficient-row orbits and term witnesses."""

    def __init__(self, partial_arity, weight_bound=2):
        self.partial_arity = ZZ(partial_arity)
        self.weight_bound = ZZ(weight_bound)
        if self.partial_arity < 1:
            raise ValueError("partial arity must be positive")
        if self.weight_bound != 2:
            raise ValueError("the registered experiment uses weights {1,2}")
        units = eisenstein_unit_actions((1, 0))
        self.alphabet = tuple(
            sorted(
                {
                    (ZZ(weight * value[0]), ZZ(weight * value[1]))
                    for weight in range(1, self.weight_bound + 1)
                    for value in units
                }
            )
        )
        if len(self.alphabet) != 12:
            raise ArithmeticError("unexpected weighted Eisenstein alphabet")
        self.sum_witnesses = self._build_sum_witnesses()
        self.local_classes = {
            support: self._build_local_classes(support)
            for support in range(self.partial_arity + 1)
        }

    def _build_sum_witnesses(self):
        tables = [{(ZZ(0), ZZ(0)): ()}]
        for depth in range(1, self.partial_arity + 1):
            table = {}
            for subtotal, witness in tables[-1].items():
                for coefficient in self.alphabet:
                    total = eisenstein_add(subtotal, coefficient)
                    candidate = tuple(sorted(witness + (coefficient,)))
                    if total not in table or candidate < table[total]:
                        table[total] = candidate
            tables.append(table)
        return tuple(tables)

    def _canonicalize_witness(self, values, witnesses):
        canonical_values, action_index = canonical_local_tuple(values)
        canonical_witnesses = tuple(
            tuple(
                sorted(
                    eisenstein_scale_by_action(term, action_index)
                    for term in coordinate_witness
                )
            )
            for coordinate_witness in witnesses
        )
        return canonical_values, canonical_witnesses

    def _build_local_classes(self, support):
        support = ZZ(support)
        zero = (ZZ(0), ZZ(0))
        if support == 0:
            witness = self.sum_witnesses[self.partial_arity].get(zero)
            return {} if witness is None else {(): (tuple(witness),)}

        classes = {}
        for allocated in range(support, self.partial_arity + 1):
            remainder = self.partial_arity - allocated
            zero_witness = self.sum_witnesses[remainder].get(zero)
            if zero_witness is None:
                continue
            for composition in positive_compositions(allocated, support):
                choices = []
                for multiplicity in composition:
                    choices.append(
                        tuple(
                            (coefficient, witness)
                            for coefficient, witness in self.sum_witnesses[multiplicity].items()
                            if coefficient != zero
                        )
                    )
                for selection in cartesian_product(choices):
                    values = tuple(item[0] for item in selection)
                    witnesses = [tuple(item[1]) for item in selection]
                    witnesses[0] = tuple(witnesses[0]) + tuple(zero_witness)
                    canonical_values, canonical_witnesses = self._canonicalize_witness(
                        values, tuple(witnesses)
                    )
                    old = classes.get(canonical_values)
                    if old is None or canonical_witnesses < old:
                        classes[canonical_values] = canonical_witnesses
        return classes

    def local_class_counts(self):
        return {
            int(support): int(len(self.local_classes[support]))
            for support in range(self.partial_arity + 1)
        }

    def global_count(self, anchor_count):
        anchor_count = ZZ(anchor_count)
        if anchor_count < self.partial_arity:
            # The formula remains valid, but this experiment only registers
            # tables with enough anchors to realize every support stratum.
            raise ValueError("anchor count must be at least the partial arity")
        return ZZ(
            sum(
                binomial(anchor_count, support)
                * len(self.local_classes[support])
                for support in range(self.partial_arity + 1)
            )
        )

    def iter_global_rows(self, anchor_count, include_witness=False):
        anchor_count = ZZ(anchor_count)
        if anchor_count < self.partial_arity:
            raise ValueError("anchor count must be at least the partial arity")
        for support in range(self.partial_arity + 1):
            classes = self.local_classes[support]
            for anchors in Combinations(range(anchor_count), support):
                for values in sorted(classes):
                    row = tuple(
                        (ZZ(anchor), ZZ(value[0]), ZZ(value[1]))
                        for anchor, value in zip(anchors, values)
                    )
                    if include_witness:
                        if support == 0:
                            witness = (
                                tuple(
                                    (ZZ(0), ZZ(term[0]), ZZ(term[1]))
                                    for term in classes[values][0]
                                ),
                            )
                        else:
                            witness = tuple(
                                tuple(
                                    (ZZ(anchor), ZZ(term[0]), ZZ(term[1]))
                                    for term in coordinate_terms
                                )
                                for anchor, coordinate_terms in zip(
                                    anchors, classes[values]
                                )
                            )
                        yield row, witness
                    else:
                        yield row

    def summary(self, anchor_count):
        anchor_count = ZZ(anchor_count)
        term_count = ZZ(len(self.alphabet) * anchor_count)
        quotient_count = self.global_count(anchor_count)
        ordered = term_count^self.partial_arity
        commutative = binomial(
            term_count + self.partial_arity - 1, self.partial_arity
        )
        return {
            "partial_arity": int(self.partial_arity),
            "relation_arity": int(self.partial_arity + 1),
            "anchor_count": int(anchor_count),
            "weighted_terms": int(term_count),
            "local_class_counts": self.local_class_counts(),
            "ordered_partial_tuples": int(ordered),
            "commutative_term_multisets": int(commutative),
            "exact_quotient_rows": int(quotient_count),
            "ordered_to_quotient_ratio": float(RR(ordered) / RR(quotient_count)),
            "commutative_to_quotient_ratio": float(
                RR(commutative) / RR(quotient_count)
            ),
        }


def eisenstein_to_field(value, fixture):
    return fixture.K(value[0]) + fixture.K(value[1]) * fixture.eigenvalue


def field_row_orbit_key(row, fixture):
    images = []
    gammas = gamma_elements(fixture)
    for gamma in gammas:
        images.append(
            tuple(
                (int(anchor), int(gamma[2] * coefficient))
                for anchor, coefficient in row
            )
        )
    return min(images)


def map_symbolic_row_to_field(row, fixture):
    mapped = tuple(
        (ZZ(anchor), eisenstein_to_field((a, b), fixture))
        for anchor, a, b in row
    )
    return field_row_orbit_key(mapped, fixture)
