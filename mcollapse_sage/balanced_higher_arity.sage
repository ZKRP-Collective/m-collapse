"""Balanced arity-four and arity-five generators over the exact pair quotient.

This SageMath module expects rank_guided_mcollapse.sage and
symmetry_quotient_mcollapse.sage to be loaded first.
"""


def transform_weighted_term(pair_index, term, gamma):
    sign, exponent, _multiplier = gamma
    atom = (
        int(term.anchor),
        int(sign * term.weight),
        int((term.exponent + exponent) % 3),
    )
    return pair_index.term_by_atom[atom]


def point_from_key(fixture, key):
    if key == (-1, -1):
        return fixture.E(0)
    return fixture.E(fixture.Fp(key[0]), fixture.Fp(key[1]))


def phase_points(fixture, point):
    positive = []
    transformed = point
    for exponent in range(3):
        positive.append(transformed)
        if exponent != 2:
            transformed = fixture.phi(transformed)
    by_label = {}
    for exponent, value in enumerate(positive):
        by_label[(1, exponent)] = value
        by_label[(-1, exponent)] = -value
    return by_label


class ExactPairEntry:
    def __init__(self, row_key, terms, point, phase_endomorphisms=0):
        self.row_key = tuple(row_key)
        self.terms = tuple(terms)
        self.point = point
        self.phase_endomorphisms = ZZ(phase_endomorphisms)


def iter_exact_pair_entries(prepared):
    pair_index = prepared.quotient_index
    fixture = prepared.fixture
    for canonical_point_key in sorted(pair_index.index):
        canonical_point = point_from_key(fixture, canonical_point_key)
        points = phase_points(fixture, canonical_point)
        phase_endomorphisms = ZZ(0 if canonical_point.is_zero() else 2)
        first_emission = True
        for left, right in pair_index.index[canonical_point_key]:
            base_row = sparse_pair_row_key(fixture, left, right)
            gammas = (
                (gamma_elements(fixture)[0],)
                if not base_row
                else gamma_elements(fixture)
            )
            for gamma in gammas:
                transformed_terms = tuple(
                    sorted(
                        (
                            transform_weighted_term(pair_index, left, gamma),
                            transform_weighted_term(pair_index, right, gamma),
                        ),
                        key=lambda term: term.atom(),
                    )
                )
                row_key = sparse_pair_row_key(
                    fixture, transformed_terms[0], transformed_terms[1]
                )
                point = points[(int(gamma[0]), int(gamma[1]))]
                yield ExactPairEntry(
                    row_key,
                    transformed_terms,
                    point,
                    phase_endomorphisms if first_emission else 0,
                )
                first_emission = False


def lift_exact_pair_entries(prepared):
    entries = tuple(iter_exact_pair_entries(prepared))
    row_keys = {entry.row_key for entry in entries}
    if len(row_keys) != len(entries):
        raise ArithmeticError("phase lifting emitted duplicate exact pair rows")
    work = zero_work()
    work["endomorphism"] = sum(
        entry.phase_endomorphisms for entry in entries
    )
    return entries, work


def sparse_terms_row_key(fixture, terms):
    row = {}
    for term in terms:
        row[term.anchor] = row.get(term.anchor, fixture.K(0)) + term.coefficient
    return tuple(
        (int(anchor), int(coefficient))
        for anchor, coefficient in sorted(row.items())
        if coefficient != 0
    )


class BalancedRelation:
    def __init__(self, terms, row_key, generation_work):
        self.terms = tuple(terms)
        self.row_key = tuple(row_key)
        self.generation_work = dict(generation_work)


class BalancedGenerationResult:
    def __init__(self, relation_arity, relations, stats, work):
        self.relation_arity = ZZ(relation_arity)
        self.relations = tuple(relations)
        self.stats = dict(stats)
        self.work = dict(work)

    def as_dict(self):
        return {
            "relation_arity": int(self.relation_arity),
            "relation_count": int(len(self.relations)),
            "stats": {name: int(value) for name, value in self.stats.items()},
            "work": work_json(self.work),
            "primitive_work": int(work_total(self.work)),
        }


class BalancedHigherArityGenerator:
    def __init__(self, prepared):
        if not isinstance(prepared, PreparedSymmetryQuotientMCollapse):
            raise TypeError("balanced generation requires the exact pair quotient")
        self.prepared = prepared
        self.fixture = prepared.fixture
        self.exact_pair_count = ZZ(
            6 * (prepared.quotient_index.representative_count - 1) + 1
        )

    def _lookup(self, point, work):
        work["table_probe"] += 1
        pairs, lookup_work = self.prepared.lookup_pairs(point)
        for field, amount in lookup_work.items():
            work[field] += amount
        return pairs

    def generate(
        self,
        target,
        relation_arity,
        max_unique=None,
        verify_relations=True,
        relation_consumer=None,
    ):
        relation_arity = ZZ(relation_arity)
        if relation_arity not in (4, 5):
            raise ValueError("balanced generator supports relation arity four or five")
        if max_unique is not None and ZZ(max_unique) < 1:
            raise ValueError("max_unique must be positive")
        max_unique = None if max_unique is None else ZZ(max_unique)
        work = zero_work()
        stats = {
            "exact_pair_rows": self.exact_pair_count,
            "complement_lookups": ZZ(0),
            "pair_hits": ZZ(0),
            "pair_order_rejections": ZZ(0),
            "duplicate_full_rows": ZZ(0),
            "unique_full_rows": ZZ(0),
            "group_failures": ZZ(0),
        }
        relations = []
        seen_rows = set()

        for first in iter_exact_pair_entries(self.prepared):
            work["endomorphism"] += first.phase_endomorphisms
            queries = (None,) if relation_arity == 4 else self.prepared.terms
            for query in queries:
                needed = target.point - first.point
                work["point_add"] += 1
                prefix_terms = first.terms
                if query is not None:
                    needed -= query.point
                    work["point_add"] += 1
                    prefix_terms = first.terms + (query,)
                stats["complement_lookups"] += 1
                second_pairs = self._lookup(needed, work)
                for second_terms in second_pairs:
                    stats["pair_hits"] += 1
                    second_key = sparse_pair_row_key(
                        self.fixture, second_terms[0], second_terms[1]
                    )
                    if second_key < first.row_key:
                        stats["pair_order_rejections"] += 1
                        continue
                    terms = tuple(
                        sorted(
                            prefix_terms + tuple(second_terms),
                            key=lambda term: term.atom(),
                        )
                    )
                    work["coefficient_op"] += relation_arity
                    row_key = sparse_terms_row_key(self.fixture, terms)
                    work["table_probe"] += 1
                    if row_key in seen_rows:
                        stats["duplicate_full_rows"] += 1
                        continue
                    seen_rows.add(row_key)
                    if verify_relations:
                        total = self.fixture.E(0)
                        for term in terms:
                            total += term.point
                        work["point_add"] += max(ZZ(0), relation_arity - 1)
                        if total != target.point:
                            stats["group_failures"] += 1
                            raise ArithmeticError("balanced relation failed group verification")
                    relation = BalancedRelation(terms, row_key, work)
                    relations.append(relation)
                    stats["unique_full_rows"] += 1
                    if relation_consumer is not None and relation_consumer(relation):
                        return BalancedGenerationResult(
                            relation_arity, relations, stats, work
                        )
                    if max_unique is not None and len(relations) >= max_unique:
                        return BalancedGenerationResult(
                            relation_arity, relations, stats, work
                        )
        return BalancedGenerationResult(relation_arity, relations, stats, work)
