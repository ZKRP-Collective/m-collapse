"""Exact S_2 x mu_6 quotient for the weighted-MCOLLAPSE arity-three pair table.

This file is SageMath code and expects rank_guided_mcollapse.sage to be loaded
first.  The quotient removes representation multiplicity before point-table
construction; it does not discard coefficient rows.
"""


def gamma_elements(fixture):
    return tuple(
        (ZZ(sign), ZZ(exponent), fixture.K(sign) * fixture.eigenvalue^exponent)
        for sign in (1, -1)
        for exponent in range(3)
    )


def gamma_point(fixture, point, gamma):
    sign, exponent, _multiplier = gamma
    transformed = fixture.phi(point, exponent)
    return transformed if sign == 1 else -transformed


def inverse_gamma(fixture, gamma):
    sign, exponent, _multiplier = gamma
    inverse_exponent = (-exponent) % 3
    return (
        sign,
        inverse_exponent,
        fixture.K(sign) * fixture.eigenvalue^inverse_exponent,
    )


def canonical_point_orientation(fixture, point):
    candidates = []
    transformed = point
    gamma_by_label = {
        (int(gamma[0]), int(gamma[1])): gamma
        for gamma in gamma_elements(fixture)
    }
    for exponent in range(3):
        for sign in (1, -1):
            candidate = transformed if sign == 1 else -transformed
            gamma = gamma_by_label[(sign, exponent)]
            candidates.append((point_key(candidate), gamma))
        if exponent != 2:
            transformed = fixture.phi(transformed)
    key, gamma = min(candidates, key=lambda item: (item[0], item[1][0], item[1][1]))
    return key, gamma


def sparse_pair_row_key(fixture, left, right):
    row = {}
    for term in (left, right):
        row[term.anchor] = row.get(term.anchor, fixture.K(0)) + term.coefficient
    return tuple(
        (int(anchor), int(coefficient))
        for anchor, coefficient in sorted(row.items())
        if coefficient != 0
    )


def scale_row_key(fixture, row_key, gamma):
    multiplier = gamma[2]
    return tuple(
        (anchor, int(multiplier * fixture.K(coefficient)))
        for anchor, coefficient in row_key
    )


def row_orbit_key(fixture, row_key):
    return min(scale_row_key(fixture, row_key, gamma) for gamma in gamma_elements(fixture))


class SymmetryQuotientPairIndex:
    """One exact representative per diagonal mu_6 orbit of unordered pairs."""

    def __init__(self, fixture, terms, representatives):
        self.fixture = fixture
        self.terms = terms
        self.representatives = representatives
        self.work = zero_work()
        self.index = {}
        self.row_orbits = set()
        self.term_by_coefficient = {}
        self.term_by_atom = {}
        for term in terms:
            key = (int(term.anchor), int(term.coefficient))
            if key in self.term_by_coefficient:
                raise ArithmeticError("weighted term coefficient is not unique")
            self.term_by_coefficient[key] = term
            atom = tuple(int(value) for value in term.atom())
            if atom in self.term_by_atom:
                raise ArithmeticError("weighted term atom is not unique")
            self.term_by_atom[atom] = term

        coefficients = sorted(
            {int(term.coefficient) for term in terms if term.anchor == 0}
        )
        if len(coefficients) != 12:
            raise ArithmeticError("the {-2,-1,1,2} x mu_3 catalog must have 12 coefficients")
        self.coefficients = tuple(coefficients)

        two_support = self._two_support_classes()
        one_support, zero_witness = self._one_support_classes()

        for left_anchor in range(len(representatives)):
            for right_anchor in range(left_anchor + 1, len(representatives)):
                for left_coefficient, right_coefficient in two_support:
                    left = self._term(left_anchor, left_coefficient)
                    right = self._term(right_anchor, right_coefficient)
                    self._register(left, right)

        for anchor in range(len(representatives)):
            for left_coefficient, right_coefficient in one_support:
                left = self._term(anchor, left_coefficient)
                right = self._term(anchor, right_coefficient)
                self._register(left, right)

        self._register(
            self._term(0, zero_witness[0]),
            self._term(0, zero_witness[1]),
        )
        for key in self.index:
            self.index[key].sort(key=lambda pair: (pair[0].atom(), pair[1].atom()))

    def _term(self, anchor, coefficient):
        return self.term_by_coefficient[(int(anchor), int(coefficient))]

    def _two_support_classes(self):
        classes = {}
        for left in self.coefficients:
            for right in self.coefficients:
                images = []
                for gamma in gamma_elements(self.fixture):
                    images.append(
                        (
                            int(gamma[2] * self.fixture.K(left)),
                            int(gamma[2] * self.fixture.K(right)),
                        )
                    )
                    self.work["coefficient_op"] += 2
                canonical = min(images)
                classes[canonical] = canonical
        if len(classes) != 24:
            raise ArithmeticError("unexpected two-support coefficient orbit count")
        return tuple(classes[key] for key in sorted(classes))

    def _one_support_classes(self):
        witness_by_sum = {}
        for left_index, left in enumerate(self.coefficients):
            for right in self.coefficients[left_index:]:
                total = self.fixture.K(left) + self.fixture.K(right)
                self.work["coefficient_op"] += 1
                key = int(total)
                witness = (left, right)
                if key not in witness_by_sum or witness < witness_by_sum[key]:
                    witness_by_sum[key] = witness

        nonzero_classes = {}
        zero_witness = witness_by_sum.get(0)
        if zero_witness is None:
            raise ArithmeticError("weighted alphabet is not closed under negation")
        for total, witness in witness_by_sum.items():
            if total == 0:
                continue
            images = []
            for gamma in gamma_elements(self.fixture):
                images.append(int(gamma[2] * self.fixture.K(total)))
                self.work["coefficient_op"] += 1
            canonical = min(images)
            canonical_witness = witness_by_sum[canonical]
            nonzero_classes[canonical] = canonical_witness
        if len(witness_by_sum) != 49 or len(nonzero_classes) != 8:
            raise ArithmeticError("unexpected one-support coefficient orbit count")
        return (
            tuple(nonzero_classes[key] for key in sorted(nonzero_classes)),
            zero_witness,
        )

    def _transform_term(self, term, gamma):
        sign, exponent, _multiplier = gamma
        atom = (
            int(term.anchor),
            int(sign * term.weight),
            int((term.exponent + exponent) % 3),
        )
        return self.term_by_atom[atom]

    def _register(self, left, right):
        raw_row = sparse_pair_row_key(self.fixture, left, right)
        orbit = row_orbit_key(self.fixture, raw_row)
        if orbit in self.row_orbits:
            raise ArithmeticError("direct quotient construction emitted a duplicate row orbit")
        self.row_orbits.add(orbit)

        point = left.point + right.point
        self.work["point_add"] += 1
        canonical_point, gamma = canonical_point_orientation(self.fixture, point)
        if not point.is_zero():
            self.work["endomorphism"] += 2
        oriented_left = self._transform_term(left, gamma)
        oriented_right = self._transform_term(right, gamma)
        if oriented_left.point + oriented_right.point != gamma_point(self.fixture, point, gamma):
            raise ArithmeticError("coefficient and point actions disagree")
        self.index.setdefault(canonical_point, []).append(
            tuple(sorted((oriented_left, oriented_right), key=lambda term: term.atom()))
        )
        self.work["table_probe"] += 1

    @property
    def representative_count(self):
        return ZZ(len(self.row_orbits))

    def lookup(self, needed):
        work = zero_work()
        canonical_point, gamma_to_canonical = canonical_point_orientation(
            self.fixture, needed
        )
        if not needed.is_zero():
            work["endomorphism"] += 2
        bucket = self.index.get(canonical_point, ())
        lifted = []
        seen_rows = set()
        if needed.is_zero():
            gammas = gamma_elements(self.fixture)
        else:
            gammas = (inverse_gamma(self.fixture, gamma_to_canonical),)
        for left, right in bucket:
            for gamma in gammas:
                transformed = tuple(
                    sorted(
                        (
                            self._transform_term(left, gamma),
                            self._transform_term(right, gamma),
                        ),
                        key=lambda term: term.atom(),
                    )
                )
                row_key = sparse_pair_row_key(
                    self.fixture, transformed[0], transformed[1]
                )
                if row_key in seen_rows:
                    continue
                seen_rows.add(row_key)
                lifted.append(transformed)
        lifted.sort(key=lambda pair: (pair[0].atom(), pair[1].atom()))
        return tuple(lifted), work


class QueryOrientedSymmetryQuotientPairIndex(SymmetryQuotientPairIndex):
    """Store direct representative points and pay for phases at lookup."""

    def _register(self, left, right):
        raw_row = sparse_pair_row_key(self.fixture, left, right)
        orbit = row_orbit_key(self.fixture, raw_row)
        if orbit in self.row_orbits:
            raise ArithmeticError("query-oriented quotient emitted a duplicate row orbit")
        self.row_orbits.add(orbit)

        point = left.point + right.point
        self.work["point_add"] += 1
        self.index.setdefault(point_key(point), []).append(
            tuple(sorted((left, right), key=lambda term: term.atom()))
        )
        self.work["table_probe"] += 1

    def lookup(self, needed):
        work = zero_work()
        positive = [needed]
        transformed = needed
        for _exponent in range(2):
            transformed = self.fixture.phi(transformed)
            positive.append(transformed)
        if not needed.is_zero():
            work["endomorphism"] += 2
        orbit = {}
        for exponent, point in enumerate(positive):
            orbit[(1, exponent)] = point
            orbit[(-1, exponent)] = -point

        lifted = []
        seen_rows = set()
        gammas = gamma_elements(self.fixture)
        for gamma in gammas:
            inverse = inverse_gamma(self.fixture, gamma)
            source = orbit[(int(inverse[0]), int(inverse[1]))]
            bucket = self.index.get(point_key(source), ())
            for left, right in bucket:
                transformed_terms = tuple(
                    sorted(
                        (
                            self._transform_term(left, gamma),
                            self._transform_term(right, gamma),
                        ),
                        key=lambda term: term.atom(),
                    )
                )
                row_key = sparse_pair_row_key(
                    self.fixture,
                    transformed_terms[0],
                    transformed_terms[1],
                )
                if row_key in seen_rows:
                    continue
                seen_rows.add(row_key)
                lifted.append(transformed_terms)
        # The caller already charges one hash lookup for this operation.
        work["table_probe"] += 5
        lifted.sort(key=lambda pair: (pair[0].atom(), pair[1].atom()))
        return tuple(lifted), work


class PreparedSymmetryQuotientMCollapse:
    def __init__(self, fixture, base_size):
        self.fixture = fixture
        self.representatives = choose_factor_base(fixture, base_size)
        self.terms, term_work = build_weighted_terms(fixture, self.representatives)
        self.quotient_index = SymmetryQuotientPairIndex(
            fixture, self.terms, self.representatives
        )
        self.preparation_work = add_work(term_work, self.quotient_index.work)

    @property
    def width(self):
        return len(self.representatives) + 1

    def lookup_pairs(self, point):
        return self.quotient_index.lookup(point)

    def as_dict(self):
        base_size = ZZ(len(self.representatives))
        predicted = 24 * binomial(base_size, 2) + 8 * base_size + 1
        return {
            "base_size": int(base_size),
            "matrix_width": int(self.width),
            "weighted_terms": int(len(self.terms)),
            "pair_index_mode": "S2-mu6-exact-quotient",
            "raw_ordered_pair_entries": int(len(self.terms)^2),
            "quotient_pair_representatives": int(
                self.quotient_index.representative_count
            ),
            "predicted_pair_representatives": int(predicted),
            "structural_compression_ratio": float(
                RR(len(self.terms)^2) / RR(self.quotient_index.representative_count)
            ),
            "preparation_work": work_json(self.preparation_work),
            "preparation_primitive_work": int(work_total(self.preparation_work)),
        }


class PreparedQueryOrientedSymmetryQuotientMCollapse(PreparedSymmetryQuotientMCollapse):
    def __init__(self, fixture, base_size):
        self.fixture = fixture
        self.representatives = choose_factor_base(fixture, base_size)
        self.terms, term_work = build_weighted_terms(fixture, self.representatives)
        self.quotient_index = QueryOrientedSymmetryQuotientPairIndex(
            fixture, self.terms, self.representatives
        )
        self.preparation_work = add_work(term_work, self.quotient_index.work)

    def as_dict(self):
        result = super().as_dict()
        result["pair_index_mode"] = "S2-mu6-query-oriented"
        return result


def exact_pair_row_sets(prepared, point):
    pairs, _work = prepared.lookup_pairs(point)
    return {
        sparse_pair_row_key(prepared.fixture, left, right)
        for left, right in pairs
    }


def unordered_pair_row_count(prepared):
    rows = set()
    for left_index, left in enumerate(prepared.terms):
        for right in prepared.terms[left_index:]:
            rows.add(sparse_pair_row_key(prepared.fixture, left, right))
    return ZZ(len(rows))


def audit_pair_lookup_equivalence(reference, quotient):
    fixture = reference.fixture
    if quotient.fixture.as_dict() != fixture.as_dict():
        raise ValueError("pair-index fixtures differ")
    mismatches = ZZ(0)
    group_failures = ZZ(0)
    compared_rows = ZZ(0)
    for scalar in range(fixture.n):
        point = scalar * fixture.G
        reference_rows = exact_pair_row_sets(reference, point)
        quotient_pairs, _work = quotient.lookup_pairs(point)
        quotient_rows = {
            sparse_pair_row_key(fixture, left, right)
            for left, right in quotient_pairs
        }
        compared_rows += len(reference_rows)
        if reference_rows != quotient_rows:
            mismatches += 1
        for left, right in quotient_pairs:
            if left.point + right.point != point:
                group_failures += 1
    return {
        "subgroup_points_checked": int(fixture.n),
        "exact_pair_rows_checked": int(compared_rows),
        "row_set_mismatches": int(mismatches),
        "group_failures": int(group_failures),
    }
