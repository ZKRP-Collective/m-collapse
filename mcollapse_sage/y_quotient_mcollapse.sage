"""Exact y-fiber pair index for j=0 GLV MCOLLAPSE fixtures."""


class YQuotientPairIndex(SymmetryQuotientPairIndex):
    """Store one row-orbit representative under y of its public pair sum."""

    def _register(self, left, right):
        raw_row = sparse_pair_row_key(self.fixture, left, right)
        orbit = row_orbit_key(self.fixture, raw_row)
        if orbit in self.row_orbits:
            raise ArithmeticError("y quotient emitted a duplicate row orbit")
        self.row_orbits.add(orbit)

        point = left.point + right.point
        self.work["point_add"] += 1
        key = None if point.is_zero() else int(point[1])
        pair = tuple(sorted((left, right), key=lambda term: term.atom()))
        pair_key = (pair[0].atom(), pair[1].atom())
        self.point_by_pair[pair_key] = point
        self.index.setdefault(key, []).append(pair)
        self.work["table_probe"] += 1

    def __init__(self, fixture, terms, representatives):
        self.point_by_pair = {}
        super().__init__(fixture, terms, representatives)

    def _lift(self, pair, gamma):
        return tuple(
            sorted(
                (
                    self._transform_term(pair[0], gamma),
                    self._transform_term(pair[1], gamma),
                ),
                key=lambda term: term.atom(),
            )
        )

    def lookup(self, needed):
        work = zero_work()
        lifted = []
        seen_rows = set()
        if needed.is_zero():
            for pair in self.index.get(None, ()):
                point = self.point_by_pair[(pair[0].atom(), pair[1].atom())]
                if not point.is_zero():
                    raise ArithmeticError("nonzero point stored in infinity bucket")
                for gamma in gamma_elements(self.fixture):
                    transformed = self._lift(pair, gamma)
                    row_key = sparse_pair_row_key(
                        self.fixture, transformed[0], transformed[1]
                    )
                    if row_key not in seen_rows:
                        seen_rows.add(row_key)
                        lifted.append(transformed)
            lifted.sort(key=lambda pair: (pair[0].atom(), pair[1].atom()))
            return tuple(lifted), work

        # The caller pays for the first lookup; charge only the second sign.
        work["table_probe"] += 1
        for sign in (ZZ(1), ZZ(-1)):
            source_y = int(self.fixture.Fp(sign) * needed[1])
            for pair in self.index.get(source_y, ()):
                source = self.point_by_pair[(pair[0].atom(), pair[1].atom())]
                transformed_point = source
                for exponent in range(3):
                    candidate = transformed_point if sign == 1 else -transformed_point
                    if candidate == needed:
                        gamma = (
                            sign,
                            ZZ(exponent),
                            self.fixture.K(sign)
                            * self.fixture.eigenvalue^exponent,
                        )
                        transformed = self._lift(pair, gamma)
                        if transformed[0].point + transformed[1].point != needed:
                            raise ArithmeticError("y quotient phase lift failed")
                        row_key = sparse_pair_row_key(
                            self.fixture, transformed[0], transformed[1]
                        )
                        if row_key not in seen_rows:
                            seen_rows.add(row_key)
                            lifted.append(transformed)
                        break
                    if exponent != 2:
                        transformed_point = self.fixture.phi(transformed_point, 1)
                        work["endomorphism"] += 1
        lifted.sort(key=lambda pair: (pair[0].atom(), pair[1].atom()))
        return tuple(lifted), work


class PreparedYQuotientMCollapse(PreparedSymmetryQuotientMCollapse):
    def __init__(self, fixture, base_size):
        self.fixture = fixture
        self.representatives = choose_factor_base(fixture, base_size)
        self.terms, term_work = build_weighted_terms(fixture, self.representatives)
        self.quotient_index = YQuotientPairIndex(
            fixture, self.terms, self.representatives
        )
        self.preparation_work = add_work(term_work, self.quotient_index.work)

    def as_dict(self):
        result = super().as_dict()
        result["pair_index_mode"] = "S2-mu6-y-fiber"
        result["y_buckets"] = int(len(self.quotient_index.index))
        return result


def audit_y_lookup_equivalence(reference, candidate):
    fixture = reference.fixture
    mismatches = ZZ(0)
    group_failures = ZZ(0)
    compared_rows = ZZ(0)
    for scalar in range(fixture.n):
        point = scalar * fixture.G
        reference_rows = exact_pair_row_sets(reference, point)
        pairs, _work = candidate.lookup_pairs(point)
        candidate_rows = {
            sparse_pair_row_key(fixture, left, right)
            for left, right in pairs
        }
        compared_rows += len(reference_rows)
        mismatches += reference_rows != candidate_rows
        for left, right in pairs:
            group_failures += left.point + right.point != point
    return {
        "subgroup_points_checked": int(fixture.n),
        "exact_pair_rows_checked": int(compared_rows),
        "row_set_mismatches": int(mismatches),
        "group_failures": int(group_failures),
    }


def predicted_pair_orbit_count(base_size):
    base_size = ZZ(base_size)
    return 24 * binomial(base_size, 2) + 8 * base_size + 1


class PreparedAdaptiveYPointMCollapse:
    """Choose the exact pair-index layout from public subgroup density."""

    def __init__(self, fixture, base_size):
        self.fixture = fixture
        self.base_size = ZZ(base_size)
        self.predicted_pair_orbits = predicted_pair_orbit_count(base_size)
        self.use_y_quotient = bool(
            fixture.n >= 3 * self.predicted_pair_orbits
        )
        selected_class = (
            PreparedYQuotientMCollapse
            if self.use_y_quotient
            else PreparedQueryOrientedSymmetryQuotientMCollapse
        )
        self.selected = selected_class(fixture, base_size)

    def __getattr__(self, name):
        return getattr(self.selected, name)

    def lookup_pairs(self, point):
        return self.selected.lookup_pairs(point)

    def as_dict(self):
        result = self.selected.as_dict()
        result["adaptive_pair_index"] = True
        result["adaptive_threshold_multiplier"] = 3
        result["predicted_pair_orbits"] = int(self.predicted_pair_orbits)
        result["selected_layout"] = (
            "y-fiber" if self.use_y_quotient else "query-oriented-point"
        )
        return result
