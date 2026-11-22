"""SageMath-native online rank filtering for weighted MCOLLAPSE.

All finite-field, elliptic-curve, vector, matrix, and rank operations in this
experiment are SageMath objects.  Python's standard library is used only for
JSON serialization in the runner.
"""

from sage.misc.prandom import random


WORK_FIELDS = (
    "endomorphism",
    "point_add",
    "point_double",
    "scalar_add",
    "scalar_double",
    "table_probe",
    "coefficient_op",
)


def zero_work():
    return {name: ZZ(0) for name in WORK_FIELDS}


def work_total(work):
    return ZZ(sum(work.values()))


def add_work(left, right):
    return {name: ZZ(left[name] + right[name]) for name in WORK_FIELDS}


def point_key(point):
    if point.is_zero():
        return (-1, -1)
    return (ZZ(point[0]), ZZ(point[1]))


class CMFixture:
    def __init__(self, label, p, b, n, generator_xy, beta, eigenvalue, seed_value):
        self.label = str(label)
        self.p = ZZ(p)
        self.b = ZZ(b)
        self.n = ZZ(n)
        self.seed = ZZ(seed_value)
        self.Fp = GF(self.p)
        self.K = GF(self.n)
        self.E = EllipticCurve(self.Fp, [0, self.Fp(self.b)])
        self.G = self.E(self.Fp(generator_xy[0]), self.Fp(generator_xy[1]))
        self.beta = self.Fp(beta)
        self.eigenvalue = self.K(eigenvalue)
        self.validate()

    def phi(self, point, exponent=1):
        result = point
        for _ in range(ZZ(exponent) % 3):
            if result.is_zero():
                return result
            result = self.E(self.beta * result[0], result[1])
        return result

    def validate(self):
        if not self.p.is_prime() or not self.n.is_prime():
            raise ValueError("fixture moduli must be prime")
        if self.p % 3 != 1 or self.beta == 1 or self.beta^3 != 1:
            raise ValueError("fixture has no nontrivial cubic endomorphism")
        if self.n * self.G != self.E(0):
            raise ValueError("fixture generator has the wrong order")
        if self.phi(self.G) != ZZ(self.eigenvalue) * self.G:
            raise ValueError("GLV eigenvalue does not match the endomorphism")
        if self.eigenvalue^2 + self.eigenvalue + 1 != 0:
            raise ValueError("GLV eigenvalue is not a cubic root of unity")

    def as_dict(self):
        return {
            "label": self.label,
            "p": int(self.p),
            "b": int(self.b),
            "n": int(self.n),
            "subgroup_bits": int(self.n.nbits()),
            "generator": [int(self.G[0]), int(self.G[1])],
            "beta": int(self.beta),
            "eigenvalue": int(self.eigenvalue),
            "seed": int(self.seed),
        }


def find_cm_fixture(subgroup_bits, seed_value):
    """Deterministically find a small j=0 prime-order subgroup in Sage."""
    subgroup_bits = ZZ(subgroup_bits)
    seed_value = ZZ(seed_value)
    if subgroup_bits < 5 or subgroup_bits > 13:
        raise ValueError("toy subgroup_bits must be in [5, 13]")

    candidate = next_prime(2^(subgroup_bits + 1) + 97 * seed_value + 1)
    for attempt in range(400):
        p = next_prime(candidate + attempt)
        if p % 3 != 1:
            continue
        Fp = GF(p)
        beta = Fp.multiplicative_generator()^((p - 1) // 3)
        if beta == 1:
            continue
        for b in range(1, min(ZZ(p), ZZ(33))):
            E = EllipticCurve(Fp, [0, Fp(b)])
            order_factors = factor(E.cardinality())
            choices = []
            for q, exponent in order_factors:
                if (
                    exponent == 1
                    and q > 3
                    and q.is_prime()
                    and ZZ(q).nbits() == subgroup_bits
                ):
                    choices.append(ZZ(q))
            for n in sorted(choices, reverse=True):
                cofactor = E.cardinality() // n
                G = None
                for point in E.points():
                    candidate_G = cofactor * point
                    if not candidate_G.is_zero() and n * candidate_G == E(0):
                        G = candidate_G
                        break
                if G is None:
                    continue
                K = GF(n)
                eigenvalues = [value for value in K if value != 1 and value^2 + value + 1 == 0]
                for eigenvalue in eigenvalues:
                    phi_G = E(beta * G[0], G[1])
                    if phi_G == ZZ(eigenvalue) * G:
                        return CMFixture(
                            "b%s-s%s" % (subgroup_bits, seed_value),
                            p,
                            b,
                            n,
                            (ZZ(G[0]), ZZ(G[1])),
                            ZZ(beta),
                            ZZ(eigenvalue),
                            seed_value,
                        )
        candidate = next_prime(p + 2)
    raise RuntimeError("could not find a deterministic toy CM fixture")


def orbit_points(fixture, point):
    points = []
    for exponent in range(3):
        transformed = fixture.phi(point, exponent)
        points.extend((transformed, -transformed))
    unique = {}
    for candidate in points:
        unique[point_key(candidate)] = candidate
    return tuple(unique[key] for key in sorted(unique))


def canonical_orbit_point(fixture, point):
    return min(orbit_points(fixture, point), key=point_key)


def choose_factor_base(fixture, base_size):
    base_size = ZZ(base_size)
    representatives = {}
    for scalar in range(1, fixture.n):
        representative = canonical_orbit_point(fixture, scalar * fixture.G)
        representatives[point_key(representative)] = representative
    ordered = tuple(representatives[key] for key in sorted(representatives))
    if len(ordered) < base_size:
        raise ValueError("requested factor base exceeds the CM orbit count")
    return ordered[:base_size]


class WeightedTerm:
    def __init__(self, point, anchor, weight, exponent, coefficient):
        self.point = point
        self.anchor = ZZ(anchor)
        self.weight = ZZ(weight)
        self.exponent = ZZ(exponent)
        self.coefficient = coefficient

    def atom(self):
        return (self.anchor, self.weight, self.exponent)


def build_weighted_terms(fixture, representatives):
    terms = []
    seen = set()
    work = zero_work()
    for anchor, representative in enumerate(representatives):
        for exponent in range(3):
            transformed = fixture.phi(representative, exponent)
            work["endomorphism"] += exponent
            for weight in (-2, -1, 1, 2):
                point = weight * transformed
                coefficient = fixture.K(weight) * fixture.eigenvalue^exponent
                key = (point_key(point), ZZ(anchor), ZZ(coefficient))
                if key in seen:
                    continue
                seen.add(key)
                if abs(weight) == 2:
                    work["point_double"] += 1
                terms.append(WeightedTerm(point, anchor, weight, exponent, coefficient))
    terms.sort(key=lambda term: (point_key(term.point), term.atom()))
    return tuple(terms), work


class PreparedWeightedMCollapse:
    def __init__(self, fixture, base_size):
        self.fixture = fixture
        self.representatives = choose_factor_base(fixture, base_size)
        self.terms, self.preparation_work = build_weighted_terms(
            fixture, self.representatives
        )
        self.pair_index = {}
        for left in self.terms:
            for right in self.terms:
                point = left.point + right.point
                self.pair_index.setdefault(point, []).append((left, right))
                self.preparation_work["point_add"] += 1
                self.preparation_work["table_probe"] += 1

    @property
    def width(self):
        return len(self.representatives) + 1

    def lookup_pairs(self, point):
        return tuple(self.pair_index.get(point, ())), zero_work()

    def as_dict(self):
        return {
            "base_size": int(len(self.representatives)),
            "matrix_width": int(self.width),
            "weighted_terms": int(len(self.terms)),
            "ordered_pair_entries": int(len(self.terms)^2),
            "preparation_work": work_json(self.preparation_work),
            "preparation_primitive_work": int(work_total(self.preparation_work)),
        }


class PublicTarget:
    def __init__(self, index, u, v, point):
        self.index = ZZ(index)
        self.u = ZZ(u)
        self.v = ZZ(v)
        self.point = point


def make_public_targets(fixture, count, seed_value):
    hidden = ZZ((131 * seed_value + 17) % (fixture.n - 1) + 1)
    Q = hidden * fixture.G
    targets = []
    seen = set()
    cursor = ZZ(0)
    while len(targets) < count:
        u = ZZ((cursor^2 + 31 * cursor + 7 * seed_value + 3) % fixture.n)
        v = ZZ((17 * cursor + 11 * seed_value + 5) % (fixture.n - 1) + 1)
        point = u * fixture.G + v * Q
        cursor += 1
        if point.is_zero():
            continue
        key = (u, v, point_key(point))
        if key in seen:
            continue
        seen.add(key)
        targets.append(PublicTarget(len(targets), u, v, point))
    return Q, tuple(targets)


class RelationCandidate:
    def __init__(self, target, terms):
        self.target = target
        self.terms = tuple(sorted(terms, key=lambda term: term.atom()))
        self.template = tuple(term.atom() for term in self.terms)
        weight_pattern = tuple(sorted(abs(term.weight) for term in self.terms))
        exponent_counts = tuple(
            sorted(
                (sum(1 for term in self.terms if term.exponent == exponent) for exponent in range(3)),
                reverse=True,
            )
        )
        self.shape = (weight_pattern, exponent_counts)


class OnlineSparseBasis:
    """Streaming echelon basis whose coefficients are Sage finite-field elements."""

    def __init__(self, field, width):
        self.field = field
        self.width = ZZ(width)
        self.pivots = {}

    @property
    def rank(self):
        return ZZ(len(self.pivots))

    def reduce(self, sparse_row):
        row = {}
        operations = ZZ(0)
        for column, value in sparse_row.items():
            column = ZZ(column)
            if column < 0 or column >= self.width:
                raise ValueError("sparse row column outside basis width")
            coefficient = self.field(value)
            operations += 1
            if coefficient != 0:
                row[column] = coefficient

        for pivot in sorted(self.pivots):
            factor_value = row.get(pivot, self.field(0))
            if factor_value == 0:
                continue
            for column, coefficient in self.pivots[pivot].items():
                updated = row.get(column, self.field(0)) - factor_value * coefficient
                operations += 2
                if updated == 0:
                    row.pop(column, None)
                else:
                    row[column] = updated

        if not row:
            return {}, operations
        pivot = min(row)
        inverse = 1 / row[pivot]
        operations += 1
        normalized = {}
        for column in sorted(row):
            value = row[column] * inverse
            operations += 1
            if value != 0:
                normalized[column] = value
        return normalized, operations

    def commit(self, residual):
        if not residual:
            raise ValueError("cannot commit a zero residual")
        pivot = min(residual)
        if residual[pivot] != 1 or pivot in self.pivots:
            raise ValueError("residual is not a new normalized pivot")
        for old_pivot in self.pivots:
            if old_pivot in residual:
                raise ValueError("residual was not reduced against the basis")
        self.pivots[pivot] = dict(residual)

    def dense_rows(self):
        rows = []
        for pivot in sorted(self.pivots):
            rows.append(vector(self.field, [self.pivots[pivot].get(i, 0) for i in range(self.width)]))
        return tuple(rows)


class OnlineDualBasis:
    """Incremental basis of the exact annihilator of the accepted row space."""

    def __init__(self, field, width, vectors=None):
        self.field = field
        self.width = ZZ(width)
        if vectors is None:
            self.vectors = [
                vector(field, [1 if column == index else 0 for column in range(width)])
                for index in range(width)
            ]
        else:
            self.vectors = [vector(field, item) for item in vectors]

    @classmethod
    def from_primal_rows(cls, field, width, rows):
        if rows:
            kernel_basis = matrix(field, rows).right_kernel().basis()
        else:
            kernel_basis = matrix(field, 0, width).right_kernel().basis()
        return cls(field, width, kernel_basis)

    @classmethod
    def from_echelon_basis(cls, primal_basis):
        field = primal_basis.field
        width = primal_basis.width
        pivot_columns = set(primal_basis.pivots)
        free_columns = [column for column in range(width) if column not in pivot_columns]
        vectors = []
        operations = ZZ(0)
        for free_column in free_columns:
            witness = vector(field, width)
            witness[free_column] = 1
            for pivot in sorted(primal_basis.pivots, reverse=True):
                total = field(0)
                for column, coefficient in primal_basis.pivots[pivot].items():
                    if column == pivot or witness[column] == 0:
                        continue
                    total += coefficient * witness[column]
                    operations += 2
                witness[pivot] = -total
                operations += 1
            vectors.append(witness)
        return cls(field, width, vectors), operations

    @property
    def nullity(self):
        return ZZ(len(self.vectors))

    @property
    def rank(self):
        return self.width - self.nullity

    def syndrome(self, sparse_row):
        values = []
        operations = ZZ(0)
        for witness in self.vectors:
            value = self.field(0)
            for column, coefficient in sparse_row.items():
                value += self.field(coefficient) * witness[column]
                operations += 2
            values.append(value)
        return tuple(values), operations

    def commit(self, sparse_row, syndrome):
        if len(syndrome) != len(self.vectors):
            raise ValueError("dual syndrome width mismatch")
        nonzero = [index for index, value in enumerate(syndrome) if value != 0]
        if not nonzero:
            raise ValueError("cannot commit a zero dual syndrome")
        witness_index = nonzero[0]
        witness = self.vectors[witness_index]
        inverse = 1 / syndrome[witness_index]
        operations = ZZ(1)
        updated = []
        for index, old_vector in enumerate(self.vectors):
            if index == witness_index:
                continue
            factor_value = syndrome[index] * inverse
            operations += 1
            new_vector = old_vector - factor_value * witness
            operations += 2 * self.width
            updated.append(new_vector)
        self.vectors = updated
        return operations

    def annihilates(self, rows):
        return all(row.dot_product(witness) == 0 for row in rows for witness in self.vectors)


def empty_stats():
    return {
        "candidates_created": ZZ(0),
        "canonicalized": ZZ(0),
        "unique": ZZ(0),
        "pivots": ZZ(0),
        "disabled": ZZ(0),
        "primitive_work": ZZ(0),
    }


def template_label(template):
    return ";".join("%s,%s,%s" % (atom[0], atom[1], atom[2]) for atom in template)


def shape_label(shape):
    return "weights=%s|glv-multiplicity=%s" % (
        ",".join(str(value) for value in shape[0]),
        ",".join(str(value) for value in shape[1]),
    )


def stats_json(stats, label_function):
    result = {}
    for key in sorted(stats, key=label_function):
        values = stats[key]
        work = values["primitive_work"]
        pivots = values["pivots"]
        result[label_function(key)] = {
            name: int(values[name]) for name in (
                "candidates_created", "canonicalized", "unique", "pivots", "disabled", "primitive_work"
            )
        }
        result[label_function(key)]["S"] = None if work == 0 else float(pivots / work)
        result[label_function(key)]["work_per_pivot"] = None if pivots == 0 else float(work / pivots)
    return result


def work_json(work):
    return {name: int(work[name]) for name in WORK_FIELDS}


class PolicyState:
    def __init__(
        self,
        prepared,
        policy,
        variable_budget,
        seed_value,
        min_samples=4,
        disable_ratio=0.30,
        hybrid_nullity_threshold=4,
    ):
        if policy not in (
            "weighted-baseline",
            "rank-filtered",
            "rank-saturated",
            "rank-guided",
            "dual-saturated",
            "hybrid-saturated",
            "adaptive-dual",
        ):
            raise ValueError("unknown policy")
        self.prepared = prepared
        self.policy = policy
        self.variable_budget = ZZ(variable_budget)
        self.seed = ZZ(seed_value)
        self.min_samples = ZZ(min_samples)
        self.disable_ratio = RR(disable_ratio)
        self.variable_work = zero_work()
        self.basis = OnlineSparseBasis(prepared.fixture.K, prepared.width)
        self.dual = (
            OnlineDualBasis(prepared.fixture.K, prepared.width)
            if policy == "dual-saturated"
            else None
        )
        self.hybrid_nullity_threshold = min(ZZ(hybrid_nullity_threshold), ZZ(prepared.width))
        if self.hybrid_nullity_threshold < 1:
            raise ValueError("hybrid nullity threshold outside matrix width")
        self.primal_tests = ZZ(0)
        self.dual_tests = ZZ(0)
        self.zero_syndromes = ZZ(0)
        self.dual_initializations = ZZ(0)
        self.adaptive_credit = ZZ(0)
        self.adaptive_observations = ZZ(0)
        self.adaptive_nullity = None
        self.adaptive_switch_rank = None
        self.last_primal_operations = ZZ(0)
        self.seen_rows = set()
        self.template_stats = {}
        self.shape_stats = {}
        self.rank_events = []
        self.candidate_trace = []
        self.pivot_trace = []
        self.verification_failures = ZZ(0)
        self.processed_targets = ZZ(0)
        self.rank_saturated_targets = ZZ(0)
        self.shared_work = ZZ(0)
        set_random_seed(self.seed)

    def spent(self):
        return work_total(self.variable_work)

    def exhausted(self):
        return self.spent() >= self.variable_budget

    def _stats(self, table, key):
        if key not in table:
            table[key] = empty_stats()
        return table[key]

    def charge(self, field, amount=1, candidate=None):
        amount = ZZ(amount)
        self.variable_work[field] += amount
        if candidate is None:
            self.shared_work += amount
            return
        exact = self._stats(self.template_stats, candidate.template)
        shape = self._stats(self.shape_stats, candidate.shape)
        exact["primitive_work"] += amount
        shape["primitive_work"] += amount

    def note(self, candidate, name, amount=1):
        amount = ZZ(amount)
        self._stats(self.template_stats, candidate.template)[name] += amount
        self._stats(self.shape_stats, candidate.shape)[name] += amount

    def shape_score(self, shape):
        values = self._stats(self.shape_stats, shape)
        samples = values["canonicalized"]
        pivots = values["pivots"]
        work = values["primitive_work"]
        probability = RR(pivots + 1) / RR(samples + 2)
        mean_cost = RR(work + 16) / RR(samples + 1)
        return probability / mean_cost

    def best_shape_score(self):
        sampled = [
            self.shape_score(shape)
            for shape, values in self.shape_stats.items()
            if values["canonicalized"] >= self.min_samples
        ]
        return max(sampled) if sampled else RR(0)

    def shape_is_disabled(self, shape):
        values = self._stats(self.shape_stats, shape)
        if values["canonicalized"] < self.min_samples:
            return False
        best = self.best_shape_score()
        return best > 0 and self.shape_score(shape) < self.disable_ratio * best

    def accept_created_candidate(self, candidate):
        self.note(candidate, "candidates_created")
        self.charge("table_probe", 1, candidate)
        if self.policy != "rank-guided" or not self.shape_is_disabled(candidate.shape):
            return True
        values = self._stats(self.shape_stats, candidate.shape)
        if values["candidates_created"] % 32 == 0:
            return True
        self.note(candidate, "disabled")
        return False

    def choose_shape(self, queues):
        shapes = sorted(queues, key=shape_label)
        weights = [max(self.shape_score(shape), RR("1e-30")) for shape in shapes]
        total = sum(weights)
        draw = RR(random()) * total
        cumulative = RR(0)
        for shape, weight in zip(shapes, weights):
            cumulative += weight
            if draw <= cumulative:
                return shape
        return shapes[-1]

    def build_raw_row(self, candidate):
        K = self.prepared.fixture.K
        raw = {}
        for term in candidate.terms:
            raw[term.anchor] = raw.get(term.anchor, K(0)) + term.coefficient
            self.charge("coefficient_op", 1, candidate)
        raw[self.prepared.width - 1] = -K(candidate.target.v)
        self.charge("coefficient_op", 1, candidate)
        return {column: value for column, value in raw.items() if value != 0}

    def normalize_raw_row(self, candidate, raw):
        K = self.prepared.fixture.K
        pivot = min(raw)
        inverse = 1 / raw[pivot]
        self.charge("coefficient_op", 1, candidate)
        normalized = {}
        for column in sorted(raw):
            normalized[column] = raw[column] * inverse
            self.charge("coefficient_op", 1, candidate)
        rhs = K(candidate.target.u) * inverse
        self.charge("coefficient_op", 1, candidate)
        return normalized, rhs

    def build_sparse_row(self, candidate):
        raw = self.build_raw_row(candidate)
        normalized, rhs = self.normalize_raw_row(candidate, raw)
        return normalized, rhs, raw

    def scalar_work(self, scalar, candidate):
        modulus = self.prepared.fixture.n
        value = ZZ(scalar) % modulus
        value = min(value, modulus - value)
        if value == 0:
            return
        bits = ZZ(value.nbits())
        self.charge("scalar_double", max(ZZ(0), bits - 1), candidate)
        self.charge("scalar_add", max(ZZ(0), ZZ(value.popcount()) - 1), candidate)

    def verify_candidate(self, candidate, raw_row):
        fixture = self.prepared.fixture
        term_total = fixture.E(0)
        for term in candidate.terms:
            term_total += term.point
        self.charge("point_add", max(ZZ(0), len(candidate.terms) - 1), candidate)

        row_total = fixture.E(0)
        nonzero = ZZ(0)
        for anchor, representative in enumerate(self.prepared.representatives):
            coefficient = raw_row.get(ZZ(anchor), fixture.K(0))
            if coefficient == 0:
                continue
            scalar = ZZ(coefficient)
            if scalar > fixture.n // 2:
                scalar -= fixture.n
            self.scalar_work(scalar, candidate)
            row_total += scalar * representative
            nonzero += 1
        self.charge("point_add", max(ZZ(0), nonzero - 1), candidate)
        valid = term_total == candidate.target.point and row_total == candidate.target.point
        if not valid:
            self.verification_failures += 1
        return valid

    def row_key(self, sparse_row):
        return tuple((int(column), int(sparse_row[column])) for column in sorted(sparse_row))

    def reduce_candidate(self, candidate, sparse_row):
        self.primal_tests += 1
        residual, operations = self.basis.reduce(sparse_row)
        self.last_primal_operations = operations
        self.charge("coefficient_op", operations, candidate)
        return residual

    def estimated_dual_initialization_work(self):
        tail_nonzeros = sum(
            max(ZZ(0), ZZ(len(row)) - 1) for row in self.basis.pivots.values()
        )
        nullity = self.prepared.width - self.basis.rank
        return nullity * (self.basis.rank + 2 * tail_nonzeros)

    def reset_adaptive_evidence(self):
        self.adaptive_credit = ZZ(0)
        self.adaptive_observations = ZZ(0)
        self.adaptive_nullity = self.prepared.width - self.basis.rank

    def observe_adaptive_dependency(self, candidate, raw_row):
        nullity = self.prepared.width - self.basis.rank
        if self.adaptive_nullity != nullity:
            self.reset_adaptive_evidence()
        primal_filter_cost = self.last_primal_operations + len(raw_row) + 2
        dual_filter_cost = 2 * len(raw_row) * nullity
        self.adaptive_credit += max(ZZ(0), ZZ(primal_filter_cost - dual_filter_cost))
        self.adaptive_observations += 1
        if (
            self.adaptive_observations >= 4
            and self.adaptive_credit >= self.estimated_dual_initialization_work()
        ):
            self.initialize_hybrid_dual(candidate, force=True)
            self.adaptive_switch_rank = self.basis.rank

    def initialize_hybrid_dual(self, candidate, force=False):
        if self.dual is not None:
            return
        nullity = self.prepared.width - self.basis.rank
        if not force and nullity > self.hybrid_nullity_threshold:
            return
        dense = self.basis.dense_rows()
        self.dual, initialization_work = OnlineDualBasis.from_echelon_basis(self.basis)
        self.charge("coefficient_op", initialization_work, candidate)
        self.dual_initializations += 1
        if self.dual.rank != self.basis.rank or not self.dual.annihilates(dense):
            raise ArithmeticError("hybrid dual initialization disagrees with Sage kernel")

    def dual_syndrome(self, candidate, raw_row):
        if self.dual is None:
            raise ArithmeticError("dual syndrome requested before initialization")
        syndrome, operations = self.dual.syndrome(raw_row)
        self.charge("coefficient_op", operations, candidate)
        self.dual_tests += 1
        if not any(syndrome):
            self.zero_syndromes += 1
        return syndrome

    def audit_primal_basis(self):
        dense = self.basis.dense_rows()
        reference_matrix = matrix(self.prepared.fixture.K, dense)
        if reference_matrix.rank() != self.basis.rank:
            raise ArithmeticError("online Sage basis disagrees with Sage matrix rank")
        if self.dual is not None:
            reference_kernel = reference_matrix.right_kernel()
            if (
                self.dual.rank != self.basis.rank
                or self.dual.nullity != reference_kernel.dimension()
                or not self.dual.annihilates(dense)
                or not all(witness in reference_kernel for witness in self.dual.vectors)
            ):
                raise ArithmeticError("dual annihilator disagrees with Sage row space")

    def record_pivot(self, candidate, residual):
        self.pivot_trace.append(
            (
                int(candidate.target.index),
                candidate.template,
                tuple(
                    (int(column), int(residual[column]))
                    for column in sorted(residual)
                ),
            )
        )
        self.basis.commit(residual)
        self.note(candidate, "pivots")
        self.rank_events.append((self.spent(), self.basis.rank))
        self.audit_primal_basis()

    def evaluate_dual_policy(self, candidate):
        raw_row = self.build_raw_row(candidate)
        use_dual = self.policy == "dual-saturated" or self.dual is not None

        if use_dual:
            syndrome = self.dual_syndrome(candidate, raw_row)
            if not any(syndrome):
                return
            self.note(candidate, "canonicalized")
            sparse_row, _normalized_rhs = self.normalize_raw_row(candidate, raw_row)
            row_key = self.row_key(sparse_row)
            if row_key in self.seen_rows:
                raise ArithmeticError("nonzero dual syndrome marked a duplicate independent")
            self.seen_rows.add(row_key)
            self.note(candidate, "unique")
            if not self.verify_candidate(candidate, raw_row):
                raise ArithmeticError("dual-filtered candidate failed exact Sage verification")
            residual, _audit_operations = self.basis.reduce(sparse_row)
            if not residual:
                raise ArithmeticError("dual syndrome disagrees with primal reduction")
            update_operations = self.dual.commit(raw_row, syndrome)
            self.charge("coefficient_op", update_operations, candidate)
            self.record_pivot(candidate, residual)
            return

        self.note(candidate, "canonicalized")
        sparse_row, _normalized_rhs = self.normalize_raw_row(candidate, raw_row)
        row_key = self.row_key(sparse_row)
        if row_key in self.seen_rows:
            return
        self.seen_rows.add(row_key)
        self.note(candidate, "unique")
        residual = self.reduce_candidate(candidate, sparse_row)
        if not residual:
            if self.policy == "adaptive-dual":
                self.observe_adaptive_dependency(candidate, raw_row)
            return
        if not self.verify_candidate(candidate, raw_row):
            raise ArithmeticError("hybrid primal candidate failed exact Sage verification")
        self.record_pivot(candidate, residual)
        if self.policy == "hybrid-saturated":
            self.initialize_hybrid_dual(candidate)
        elif self.policy == "adaptive-dual":
            self.reset_adaptive_evidence()

    def evaluate(self, candidate):
        if self.policy in ("dual-saturated", "hybrid-saturated", "adaptive-dual"):
            self.evaluate_dual_policy(candidate)
            return
        raw_row = self.build_raw_row(candidate)
        self.evaluate_prebuilt_raw(candidate, raw_row)

    def evaluate_prebuilt_raw(self, candidate, raw_row):
        """Evaluate a row whose term coefficients were charged upstream."""
        self.candidate_trace.append(
            (int(candidate.target.index), candidate.template)
        )
        self.note(candidate, "canonicalized")
        sparse_row, _normalized_rhs = self.normalize_raw_row(candidate, raw_row)
        row_key = self.row_key(sparse_row)

        if self.policy == "weighted-baseline":
            if not self.verify_candidate(candidate, raw_row):
                raise ArithmeticError("pair-index candidate failed exact Sage verification")
            if row_key in self.seen_rows:
                return
            self.seen_rows.add(row_key)
            self.note(candidate, "unique")
            residual = self.reduce_candidate(candidate, sparse_row)
        else:
            if row_key in self.seen_rows:
                return
            self.seen_rows.add(row_key)
            self.note(candidate, "unique")
            residual = self.reduce_candidate(candidate, sparse_row)
            if not residual:
                return
            if not self.verify_candidate(candidate, raw_row):
                raise ArithmeticError("rank-filtered candidate failed exact Sage verification")

        if not residual:
            return
        self.record_pivot(candidate, residual)

    def process_window(self, window, stop_rank=None):
        if self.policy != "rank-guided":
            for candidate in window:
                if self.exhausted():
                    return
                self.evaluate(candidate)
                if stop_rank is not None and self.basis.rank >= stop_rank:
                    return
            return

        queues = {}
        for candidate in window:
            queues.setdefault(candidate.shape, []).append(candidate)
        while queues and not self.exhausted():
            shape = self.choose_shape(queues)
            candidate = queues[shape].pop(0)
            if not queues[shape]:
                del queues[shape]
            self.evaluate(candidate)
            if stop_rank is not None and self.basis.rank >= stop_rank:
                return

    def rank_at(self, frontier):
        frontier = ZZ(frontier)
        result = ZZ(0)
        for event_work, event_rank in self.rank_events:
            if event_work > frontier:
                break
            result = event_rank
        return result

    def result(self, frontiers):
        preparation = self.prepared.preparation_work
        total = add_work(preparation, self.variable_work)
        frontier_rows = []
        for frontier in frontiers:
            rank_value = self.rank_at(frontier)
            denominator = work_total(preparation) + frontier
            frontier_rows.append({
                "variable_work_frontier": int(frontier),
                "total_work_frontier": int(denominator),
                "independent_relations": int(rank_value),
                "independent_relations_per_primitive": float(RR(rank_value) / RR(denominator)),
            })
        full_rank_work = None
        for event_work, event_rank in self.rank_events:
            if event_rank == self.prepared.width:
                full_rank_work = work_total(preparation) + event_work
                break
        return {
            "policy": self.policy,
            "rank": int(self.basis.rank),
            "matrix_width": int(self.prepared.width),
            "processed_targets": int(self.processed_targets),
            "rank_saturated_targets": int(self.rank_saturated_targets),
            "primal_tests": int(self.primal_tests),
            "dual_tests": int(self.dual_tests),
            "zero_syndromes": int(self.zero_syndromes),
            "dual_initializations": int(self.dual_initializations),
            "final_dual_nullity": None if self.dual is None else int(self.dual.nullity),
            "hybrid_nullity_threshold": int(self.hybrid_nullity_threshold),
            "adaptive_credit": int(self.adaptive_credit),
            "adaptive_observations": int(self.adaptive_observations),
            "adaptive_switch_rank": None if self.adaptive_switch_rank is None else int(self.adaptive_switch_rank),
            "unique_rows": int(len(self.seen_rows)),
            "verification_failures": int(self.verification_failures),
            "work": work_json(total),
            "primitive_work": int(work_total(total)),
            "variable_primitive_work": int(self.spent()),
            "shared_variable_work": int(self.shared_work),
            "work_to_full_rank": None if full_rank_work is None else int(full_rank_work),
            "rank_events": [[int(event_work), int(event_rank)] for event_work, event_rank in self.rank_events],
            "frontiers": frontier_rows,
            "template_stats": stats_json(self.template_stats, template_label),
            "shape_stats": stats_json(self.shape_stats, shape_label),
        }


def run_policy(
    prepared,
    targets,
    policy,
    variable_budget,
    seed_value,
    window_size=None,
    hybrid_nullity_threshold=4,
):
    if window_size is None:
        window_size = 96 if policy == "rank-guided" else 1
    window_size = ZZ(window_size)
    if window_size < 1:
        raise ValueError("window size must be positive")
    state = PolicyState(
        prepared,
        policy,
        variable_budget,
        seed_value,
        hybrid_nullity_threshold=hybrid_nullity_threshold,
    )
    fixture = prepared.fixture
    for target in targets:
        if state.exhausted() or state.basis.rank == prepared.width:
            break
        state.processed_targets += 1
        target_start_rank = state.basis.rank
        target_saturated = False
        window = []
        for query in prepared.terms:
            if state.exhausted() or state.basis.rank == prepared.width:
                break
            needed = target.point - query.point
            state.charge("point_add", 1)
            state.charge("table_probe", 1)
            pair_hits, lookup_work = prepared.lookup_pairs(needed)
            for field, amount in lookup_work.items():
                state.charge(field, amount)
            for left, right in pair_hits:
                if state.exhausted() or state.basis.rank == prepared.width:
                    break
                candidate = RelationCandidate(target, (left, right, query))
                if state.accept_created_candidate(candidate):
                    window.append(candidate)
                if len(window) >= window_size:
                    stop_rank = None
                    if policy in (
                        "rank-saturated",
                        "rank-guided",
                        "dual-saturated",
                        "hybrid-saturated",
                        "adaptive-dual",
                    ):
                        stop_rank = min(
                            prepared.width,
                            target_start_rank + len(prepared.representatives),
                        )
                    state.process_window(window, stop_rank=stop_rank)
                    window = []
                    if state.basis.rank == prepared.width:
                        target_saturated = True
                        break
                    if (
                        policy in (
                            "rank-saturated",
                            "rank-guided",
                            "dual-saturated",
                            "hybrid-saturated",
                            "adaptive-dual",
                        )
                        and state.basis.rank - target_start_rank >= len(prepared.representatives)
                    ):
                        state.rank_saturated_targets += 1
                        target_saturated = True
                        break
                    if state.exhausted():
                        break
            if target_saturated:
                break
        if window and not state.exhausted() and state.basis.rank < prepared.width:
            state.process_window(window)
    return state


def compare_policies(
    fixture,
    base_size,
    target_count,
    variable_budget,
    frontiers,
    seed_value,
    primary_frontier=None,
):
    prepared = PreparedWeightedMCollapse(fixture, base_size)
    Q, targets = make_public_targets(fixture, target_count, seed_value)
    for target in targets:
        if target.point != target.u * fixture.G + target.v * Q:
            raise ArithmeticError("public target construction failed")

    results = {}
    for offset, policy in enumerate(
        ("weighted-baseline", "rank-filtered", "rank-saturated", "rank-guided")
    ):
        state = run_policy(
            prepared,
            targets,
            policy,
            variable_budget,
            ZZ(seed_value) + 1009 * offset,
        )
        results[policy] = state.result(frontiers)

    primary = ZZ(frontiers[-1] if primary_frontier is None else primary_frontier)
    if primary not in frontiers:
        raise ValueError("primary frontier must be one of the recorded frontiers")
    primary_index = tuple(frontiers).index(primary)
    baseline_row = results["weighted-baseline"]["frontiers"][primary_index]
    guided_row = results["rank-guided"]["frontiers"][primary_index]
    baseline_yield = RR(baseline_row["independent_relations_per_primitive"])
    guided_yield = RR(guided_row["independent_relations_per_primitive"])
    filtered_row = results["rank-filtered"]["frontiers"][primary_index]
    filtered_yield = RR(filtered_row["independent_relations_per_primitive"])
    saturated_row = results["rank-saturated"]["frontiers"][primary_index]
    saturated_yield = RR(saturated_row["independent_relations_per_primitive"])
    ratio = None if baseline_yield == 0 else guided_yield / baseline_yield
    filtered_ratio = None if baseline_yield == 0 else filtered_yield / baseline_yield
    guided_to_saturated = None if saturated_yield == 0 else guided_yield / saturated_yield
    return {
        "fixture": fixture.as_dict(),
        "prepared": prepared.as_dict(),
        "target_count": int(target_count),
        "variable_work_budget": int(variable_budget),
        "primary_variable_frontier": int(primary),
        "policies": results,
        "primary_guided_to_baseline_yield_ratio": None if ratio is None else float(ratio),
        "primary_filtered_to_baseline_yield_ratio": None if filtered_ratio is None else float(filtered_ratio),
        "primary_guided_to_saturated_yield_ratio": None if guided_to_saturated is None else float(guided_to_saturated),
    }
