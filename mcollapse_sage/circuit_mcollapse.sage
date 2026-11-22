"""Circuit-first MCOLLAPSE: eliminate anchor logs and solve the target directly."""


class OnlineAnchorCircuitBasis:
    def __init__(self, field, width):
        self.field = field
        self.width = ZZ(width)
        self.pivots = {}

    @property
    def rank(self):
        return ZZ(len(self.pivots))

    def reduce(self, anchor_row, q_coefficient, rhs):
        row = {}
        operations = ZZ(0)
        for column, value in anchor_row.items():
            coefficient = self.field(value)
            operations += 1
            if coefficient:
                row[ZZ(column)] = coefficient
        q_coefficient = self.field(q_coefficient)
        rhs = self.field(rhs)
        operations += 2

        for pivot in sorted(self.pivots):
            factor_value = row.get(pivot, self.field(0))
            if not factor_value:
                continue
            pivot_row, pivot_q, pivot_rhs = self.pivots[pivot]
            for column, coefficient in pivot_row.items():
                updated = row.get(column, self.field(0)) - factor_value * coefficient
                operations += 2
                if updated:
                    row[column] = updated
                else:
                    row.pop(column, None)
            q_coefficient -= factor_value * pivot_q
            rhs -= factor_value * pivot_rhs
            operations += 4

        if not row:
            return {}, q_coefficient, rhs, operations
        pivot = min(row)
        inverse = 1 / row[pivot]
        operations += 1
        normalized = {}
        for column in sorted(row):
            normalized[column] = row[column] * inverse
            operations += 1
        q_coefficient *= inverse
        rhs *= inverse
        operations += 2
        return normalized, q_coefficient, rhs, operations

    def commit(self, row, q_coefficient, rhs):
        if not row:
            raise ValueError("cannot commit an anchor dependency")
        pivot = min(row)
        if pivot in self.pivots or row[pivot] != 1:
            raise ValueError("invalid normalized circuit pivot")
        self.pivots[pivot] = (dict(row), q_coefficient, rhs)


def iter_target_candidate_events(prepared, target):
    pending = zero_work()
    for query in prepared.terms:
        needed = target.point - query.point
        pending["point_add"] += 1
        pending["table_probe"] += 1
        pair_hits, lookup_work = prepared.lookup_pairs(needed)
        for field, amount in lookup_work.items():
            pending[field] += amount
        for left, right in pair_hits:
            pending["table_probe"] += 1
            candidate = RelationCandidate(target, (left, right, query))
            yield candidate, dict(pending)
            pending = zero_work()


def make_distinct_random_public_targets(fixture, count, seed_value):
    """Make Sage-random public combinations with distinct nonzero points."""
    count = ZZ(count)
    if count < 1 or count >= fixture.n:
        raise ValueError("target count must be smaller than the subgroup")
    set_random_seed(ZZ(seed_value))
    hidden = ZZ.random_element(1, fixture.n)
    Q = hidden * fixture.G
    targets = []
    seen_points = set()
    while len(targets) < count:
        u = ZZ.random_element(fixture.n)
        v = ZZ.random_element(1, fixture.n)
        point = u * fixture.G + v * Q
        key = point_key(point)
        if point.is_zero() or key in seen_points:
            continue
        seen_points.add(key)
        targets.append(PublicTarget(len(targets), u, v, point))
    return Q, tuple(targets)


class CircuitMCollapseResult:
    def __init__(
        self,
        solved,
        recovered,
        state,
        basis,
        relations,
        dependencies,
        equations,
    ):
        self.solved = bool(solved)
        self.recovered = recovered
        self.state = state
        self.basis = basis
        self.relations = ZZ(relations)
        self.dependencies = ZZ(dependencies)
        self.equations = tuple(equations)

    def exact_rank_audit(self):
        K = self.state.prepared.fixture.K
        anchor_width = ZZ(self.basis.width)
        anchor_rows = []
        coefficient_rows = []
        augmented_rows = []
        for anchor_row, q_coefficient, rhs in self.equations:
            dense_anchor = [K(anchor_row.get(column, 0)) for column in range(anchor_width)]
            anchor_rows.append(dense_anchor)
            coefficient_rows.append(dense_anchor + [K(q_coefficient)])
            augmented_rows.append(dense_anchor + [K(q_coefficient), K(rhs)])
        anchor_rank = matrix(K, anchor_rows).rank() if anchor_rows else ZZ(0)
        coefficient_rank = (
            matrix(K, coefficient_rows).rank() if coefficient_rows else ZZ(0)
        )
        augmented_rank = (
            matrix(K, augmented_rows).rank() if augmented_rows else ZZ(0)
        )
        return {
            "anchor_matrix_rank": int(anchor_rank),
            "coefficient_matrix_rank": int(coefficient_rank),
            "rhs_augmented_matrix_rank": int(augmented_rank),
            "online_anchor_rank_matches": bool(anchor_rank == self.basis.rank),
            "system_consistent": bool(augmented_rank == coefficient_rank),
            "target_column_adds_rank": bool(coefficient_rank == anchor_rank + 1),
        }

    def as_dict(self):
        result = {
            "solved": self.solved,
            "recovered_scalar": None if self.recovered is None else int(self.recovered),
            "anchor_rank": int(self.basis.rank),
            "anchor_width": int(self.basis.width),
            "relations_processed": int(self.relations),
            "anchor_dependencies": int(self.dependencies),
            "processed_targets": int(self.state.processed_targets),
            "verification_failures": int(self.state.verification_failures),
            "variable_work": int(self.state.spent()),
            "total_with_preparation": int(
                work_total(self.state.prepared.preparation_work) + self.state.spent()
            ),
            "work": work_json(
                add_work(self.state.prepared.preparation_work, self.state.variable_work)
            ),
        }
        result["exact_rank_audit"] = self.exact_rank_audit()
        return result


def run_circuit_mcollapse(
    prepared,
    targets,
    variable_budget,
    seed_value,
    round_robin=True,
):
    state = PolicyState(prepared, "rank-saturated", variable_budget, seed_value)
    basis = OnlineAnchorCircuitBasis(
        prepared.fixture.K, len(prepared.representatives)
    )
    streams = [iter_target_candidate_events(prepared, target) for target in targets]
    active = list(range(len(streams)))
    relations = ZZ(0)
    dependencies = ZZ(0)
    recovered = None
    equations = []

    while active and not state.exhausted() and recovered is None:
        schedule = tuple(active) if round_robin else (active[0],)
        for stream_index in schedule:
            if stream_index not in active or state.exhausted():
                continue
            try:
                candidate, generation_work = next(streams[stream_index])
            except StopIteration:
                active.remove(stream_index)
                continue
            state.processed_targets = max(
                state.processed_targets, ZZ(stream_index + 1)
            )
            for field, amount in generation_work.items():
                state.charge(field, amount)
            state.note(candidate, "candidates_created")
            state.note(candidate, "canonicalized")
            anchor_row = {}
            for term in candidate.terms:
                anchor_row[term.anchor] = (
                    anchor_row.get(term.anchor, prepared.fixture.K(0))
                    + term.coefficient
                )
                state.charge("coefficient_op", 1, candidate)
            anchor_row = {
                column: coefficient
                for column, coefficient in anchor_row.items()
                if coefficient
            }
            equations.append(
                (
                    dict(anchor_row),
                    -prepared.fixture.K(candidate.target.v),
                    prepared.fixture.K(candidate.target.u),
                )
            )
            state.candidate_trace.append(
                (int(candidate.target.index), candidate.template)
            )
            residual, q_coefficient, rhs, reduction_work = basis.reduce(
                anchor_row,
                -prepared.fixture.K(candidate.target.v),
                prepared.fixture.K(candidate.target.u),
            )
            state.charge("coefficient_op", reduction_work, candidate)
            relations += 1
            if residual:
                if not state.verify_candidate(candidate, anchor_row):
                    raise ArithmeticError("circuit pivot failed exact verification")
                basis.commit(residual, q_coefficient, rhs)
                state.note(candidate, "pivots")
                continue
            dependencies += 1
            if q_coefficient == 0:
                if rhs != 0:
                    raise ArithmeticError("inconsistent public MCOLLAPSE circuit")
                continue
            if not state.verify_candidate(candidate, anchor_row):
                raise ArithmeticError("solving circuit failed exact verification")
            recovered = rhs / q_coefficient
            state.charge("coefficient_op", 2, candidate)
            break

    return CircuitMCollapseResult(
        recovered is not None,
        recovered,
        state,
        basis,
        relations,
        dependencies,
        equations,
    )
