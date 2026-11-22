load("mcollapse_sage/structured_circuit_spectrum.sage")


def assert_raises_value_error(callback):
    try:
        callback()
    except ValueError:
        return
    raise AssertionError("expected ValueError")


def test_frozen_contract_literals():
    assert M_VALUES == (4, 6, 8, 10, 12)
    assert STRUCTURED_FIXTURES == (
        (11, 1601, 1471),
        (11, 1607, 1597),
        (12, 1609, 3823),
        (12, 1613, 2203),
        (13, 1619, 4831),
        (13, 1621, 6421),
    )
    assert TARGET_COUNT == 32
    assert CONTROL_TRIALS == 32
    assert STRUCTURED_CONTROLS == (
        "mcollapse_round_robin",
        "within_target_shuffle",
        "global_shuffle",
        "matched_sparse_control",
    )
    assert TARGET_V_MODE == "one_immutable_v_per_public_target"
    assert CENSOR_VALUE == "stream_length+1"
    assert VERDICT_RETAIN == "RETAIN_STRUCTURED_CIRCUIT_SURPLUS"
    assert VERDICT_REJECT == "REJECT_STRUCTURED_CIRCUIT_SURPLUS"
    assert M12_REQUIRED_WINS == 5
    assert M12_TOTAL_FIXTURES == 6
    assert MEDIAN_RATIO_CEILING == QQ(90) / QQ(100)
    assert REQUIRED_SLOPE_SIGN == "negative"


def test_reference_order_is_not_a_comparator_gate():
    assert STRUCTURED_EVALUATED_ORDERS_AND_NULLS == (
        "mcollapse_round_robin",
        "within_target_shuffle",
        "global_shuffle",
        "matched_sparse_control",
    )
    assert STRUCTURED_COMPARATORS == (
        "within_target_shuffle",
        "global_shuffle",
        "matched_sparse_control",
    )
    assert "mcollapse_round_robin" not in STRUCTURED_COMPARATORS
    assert set(STRUCTURED_COMPARATORS).issubset(set(STRUCTURED_EVALUATED_ORDERS_AND_NULLS))


def test_contract_validators_reject_malformed_inputs():
    assert_raises_value_error(
        lambda: make_spectrum_contract(
            fixtures=((11, 1601, 1471), (12, 1601, 3823))
        )
    )
    assert_raises_value_error(
        lambda: make_spectrum_contract(
            fixtures=((11, 1601, 1471), (11, 1601, 1471))
        )
    )
    assert_raises_value_error(
        lambda: make_spectrum_contract(controls=("mcollapse_round_robin", "bogus"))
    )
    assert_raises_value_error(lambda: make_spectrum_contract(target_count=0))
    assert_raises_value_error(lambda: make_spectrum_contract(control_trials=0))
    assert_raises_value_error(lambda: validate_target_v_mode("per_relation_v"))


def test_contract_rejects_positive_but_non_frozen_scientific_fields():
    assert_raises_value_error(
        lambda: make_spectrum_contract(
            fixtures=(
                (11, 1601, 1471),
                (11, 1607, 1597),
                (12, 1609, 3823),
                (12, 1613, 2203),
                (13, 1619, 4831),
                (13, 1627, 6421),
            )
        )
    )
    assert_raises_value_error(lambda: make_spectrum_contract(m_values=(4,)))
    assert_raises_value_error(lambda: make_spectrum_contract(target_count=31))
    assert_raises_value_error(lambda: make_spectrum_contract(control_trials=1))
    assert_raises_value_error(lambda: make_spectrum_contract(controls=STRUCTURED_COMPARATORS))
    assert_raises_value_error(
        lambda: make_spectrum_contract(
            controls=(
                "within_target_shuffle",
                "mcollapse_round_robin",
                "global_shuffle",
                "matched_sparse_control",
            )
        )
    )


def test_seed_derivation_is_sha256_and_stable():
    expected = "structured-circuit-spectrum|fixture_seed=1601|m=12|control=global_shuffle|trial=7"
    assert seed_material(1601, 12, "global_shuffle", 7) == expected
    assert derive_trial_seed(1601, 12, "global_shuffle", 7) == int(
        "1d273eac74598588ad23331e7ba49503e37925f40c12cc7f576d9191df4e703f",
        16,
    )
    assert_raises_value_error(lambda: derive_trial_seed(1601, 12, "unknown", 0))


def test_raw_schema_and_gate_math_are_frozen():
    assert RAW_RECORD_FIELDS == (
        "schema_version",
        "fixture_bits",
        "fixture_seed",
        "fixture_order",
        "m",
        "relation_id",
        "target_index",
        "target_u",
        "target_v",
        "target_point",
        "anchor_row",
        "q",
        "rhs",
        "term_atoms",
        "term_points",
        "support_key",
        "normalized_anchor_key",
        "normalized_full_equation_key",
        "generation_ledger",
    )
    assert CORRECTNESS_AUDITS == (
        "relation_point_audit",
        "duplicate_row_audit",
        "support_repetition_audit",
        "target_mixing_audit",
        "carried_coefficient_audit",
        "recovery_audit",
        "rank_audit",
        "point_witness_audit",
        "fixed_target_v_audit",
        "raw_schema_audit",
        "control_histogram_audit",
        "deterministic_seed_audit",
    )
    assert exact_median((9, 1, 5)) == QQ(5)
    assert exact_median((1, 9, 5, 7)) == QQ(6)
    assert censor_relations(10, False, 99) == 10
    assert censor_relations(None, True, 99) == 100
    assert abs(least_squares_log2_slope((4, 8, 16), (QQ(1), QQ(1) / QQ(2), QQ(1) / QQ(4))) + 1) < 1e-12
    assert gate_thresholds_as_dict() == {
        "m12_wins": "5/6",
        "median_ratio_ceiling": "0.90",
        "required_slope": "<0",
    }


def make_toy_prepared():
    fixture = find_cm_fixture(7, 11)
    prepared = PreparedWeightedMCollapse(fixture, 4)
    Q, targets = make_distinct_random_public_targets(fixture, 3, 2026)
    return fixture, prepared, Q, targets


def test_relation_stream_records_and_audits_are_exact():
    fixture, prepared, _Q, targets = make_toy_prepared()
    records = collect_structured_relation_stream(prepared, targets)
    first_recovery = collect_until_first_useful(prepared, targets)
    assert len(records) > first_recovery["relations_processed"]
    assert tuple(records[0].as_ordered_dict().keys()) == RAW_RECORD_FIELDS
    assert all(audit_relation_record(prepared, record) for record in records)
    assert audit_fixed_targets(records)
    assert audit_candidate_sequence_equivalence(prepared, prepared, targets)
    broken = records[0].with_updates(target_v=records[0].target_v + 1)
    assert_raises_value_error(lambda: audit_relation_record(prepared, broken, strict=True))


def test_provenance_basis_covers_pivot_and_circuit_cases():
    K = GF(101)
    rows = [
        make_toy_relation(K, 0, {0: 1}, -2, 7, 0, (0,)),
        make_toy_relation(K, 1, {1: 1}, 0, 5, 1, (1,)),
        make_toy_relation(K, 2, {0: 1, 1: 1}, -2, 12, 2, (0, 1)),
        make_toy_relation(K, 3, {0: 1, 1: 1}, -2, 12, 2, (0, 1)),
        make_toy_relation(K, 4, {0: 2, 1: 2}, -4, 24, 1, (0, 1)),
    ]
    spectrum = enumerate_provenance_circuits(K, 2, rows)
    assert spectrum.pivot_count == 2
    assert [w.dependency_length for w in spectrum.witnesses] == [3, 2, 3]
    assert spectrum.witnesses[0].useful and spectrum.witnesses[0].recovered == K(6)
    assert not spectrum.witnesses[1].target_mixing
    assert spectrum.witnesses[1].duplicate_row
    assert spectrum.witnesses[2].repeated_support
    assert all(audit_circuit_witness(K, 2, rows, witness) for witness in spectrum.witnesses)
    tampered = spectrum.witnesses[0].with_updates(source_coefficients={0: K(1)})
    assert_raises_value_error(lambda: audit_circuit_witness(K, 2, rows, tampered, strict=True))


def test_deterministic_controls_and_sparse_null_are_exact():
    fixture, prepared, _Q, targets = make_toy_prepared()
    records = collect_structured_relation_stream(prepared, targets[:2])
    assert len({derive_trial_seed(1601, 4, "within_target_shuffle", trial) for trial in range(32)}) == 32
    within_a = schedule_control_records(records, "within_target_shuffle", 1601, 4, 3)
    within_b = schedule_control_records(tuple(reversed(records)), "within_target_shuffle", 1601, 4, 3)
    global_a = schedule_control_records(records, "global_shuffle", 1601, 4, 3)
    assert order_hash(within_a) == order_hash(schedule_control_records(records, "within_target_shuffle", 1601, 4, 3))
    assert sorted(record.relation_id for record in within_a) == sorted(record.relation_id for record in records)
    assert sorted(record.relation_id for record in within_b) == sorted(record.relation_id for record in records)
    assert sorted(record.relation_id for record in global_a) == sorted(record.relation_id for record in records)
    assert all((record.target_u, record.target_v, record.target_point) == target_tuple_by_index(records)[record.target_index] for record in global_a)
    null_result = build_matched_sparse_null(prepared, records, 1601, 4, 5)
    assert null_result.valid
    assert audit_null_histograms(records, null_result.records)
    assert all(audit_relation_record(prepared, record) for record in null_result.records)
    impossible = records[0].with_updates(support_key=(999,))
    invalid = build_matched_sparse_null(prepared, (impossible,), 1601, 4, 5)
    assert not invalid.valid and invalid.reason


test_frozen_contract_literals()
test_reference_order_is_not_a_comparator_gate()
test_contract_validators_reject_malformed_inputs()
test_contract_rejects_positive_but_non_frozen_scientific_fields()
test_seed_derivation_is_sha256_and_stable()
test_raw_schema_and_gate_math_are_frozen()
test_relation_stream_records_and_audits_are_exact()
test_provenance_basis_covers_pivot_and_circuit_cases()
test_deterministic_controls_and_sparse_null_are_exact()
print("structured_circuit_spectrum Sage tests: PASS")
