"""Equal-work online rank runner for exact balanced higher-arity MCOLLAPSE.

This module expects rank_guided_mcollapse.sage, symmetry_quotient_mcollapse.sage, and
balanced_higher_arity.sage to be loaded first.
"""


def charge_generation_delta(state, previous, current):
    for field in WORK_FIELDS:
        delta = ZZ(current[field]) - ZZ(previous[field])
        if delta < 0:
            raise ArithmeticError("balanced generation ledger moved backwards")
        if delta:
            state.charge(field, delta)


def add_stats(target, source):
    for name, value in source.items():
        target[name] = target.get(name, ZZ(0)) + ZZ(value)


class BalancedEqualWorkRun:
    def __init__(self, state, relation_arity, generation_stats, target_ledgers):
        self.state = state
        self.relation_arity = ZZ(relation_arity)
        self.generation_stats = dict(generation_stats)
        self.target_ledgers = tuple(target_ledgers)

    def result(self, frontiers):
        result = self.state.result(frontiers)
        result["relation_arity"] = int(self.relation_arity)
        result["generation_stats"] = {
            name: int(value) for name, value in self.generation_stats.items()
        }
        result["target_ledgers"] = tuple(self.target_ledgers)
        return result


def run_balanced_equal_work(
    prepared,
    targets,
    relation_arity,
    variable_budget,
    seed_value,
):
    relation_arity = ZZ(relation_arity)
    if relation_arity not in (4, 5):
        raise ValueError("equal-work runner supports arity four or five")
    state = PolicyState(
        prepared,
        "rank-saturated",
        variable_budget,
        seed_value,
    )
    generator = BalancedHigherArityGenerator(prepared)
    aggregate_stats = {}
    target_ledgers = []

    for target in targets:
        if state.exhausted() or state.basis.rank == prepared.width:
            break
        state.processed_targets += 1
        start_rank = state.basis.rank
        stop_rank = min(
            prepared.width,
            start_rank + len(prepared.representatives),
        )
        previous_generation = [zero_work()]
        stopped = [False]

        def consume(relation):
            charge_generation_delta(
                state,
                previous_generation[0],
                relation.generation_work,
            )
            previous_generation[0] = dict(relation.generation_work)
            candidate = RelationCandidate(target, relation.terms)
            state.note(candidate, "candidates_created")
            raw_row = {
                ZZ(anchor): prepared.fixture.K(coefficient)
                for anchor, coefficient in relation.row_key
            }
            raw_row[prepared.width - 1] = -prepared.fixture.K(target.v)
            state.charge("coefficient_op", 1, candidate)
            state.evaluate_prebuilt_raw(candidate, raw_row)
            stopped[0] = state.basis.rank >= stop_rank or state.exhausted()
            return stopped[0]

        generated = generator.generate(
            target,
            relation_arity,
            verify_relations=False,
            relation_consumer=consume,
        )
        if not stopped[0]:
            charge_generation_delta(
                state,
                previous_generation[0],
                generated.work,
            )
        add_stats(aggregate_stats, generated.stats)
        target_ledgers.append({
            "target_index": int(target.index),
            "start_rank": int(start_rank),
            "end_rank": int(state.basis.rank),
            "stop_rank": int(stop_rank),
            "relations_emitted": int(len(generated.relations)),
            "generation_work": int(work_total(generated.work)),
            "complement_lookups": int(generated.stats["complement_lookups"]),
        })
        if state.basis.rank >= stop_rank and state.basis.rank < prepared.width:
            state.rank_saturated_targets += 1

    return BalancedEqualWorkRun(
        state,
        relation_arity,
        aggregate_stats,
        target_ledgers,
    )
