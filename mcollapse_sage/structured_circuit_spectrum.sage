import hashlib


M_VALUES = (4, 6, 8, 10, 12)
STRUCTURED_FIXTURES = (
    (11, 1601, 1471),
    (11, 1607, 1597),
    (12, 1609, 3823),
    (12, 1613, 2203),
    (13, 1619, 4831),
    (13, 1621, 6421),
)
TARGET_COUNT = 32
CONTROL_TRIALS = 32
STRUCTURED_EVALUATED_ORDERS_AND_NULLS = (
    "mcollapse_round_robin",
    "within_target_shuffle",
    "global_shuffle",
    "matched_sparse_control",
)
STRUCTURED_CONTROLS = STRUCTURED_EVALUATED_ORDERS_AND_NULLS
STRUCTURED_COMPARATORS = (
    "within_target_shuffle",
    "global_shuffle",
    "matched_sparse_control",
)
TARGET_V_MODE = "one_immutable_v_per_public_target"
REJECTED_TARGET_V_MODES = ("per_relation_v",)
TRIAL_SEED_HASH = "sha256"
SEED_DOMAIN = "structured-circuit-spectrum"
CENSOR_VALUE = "stream_length+1"
MEDIAN_FORMULA = "exact_median"
SLOPE_FORMULA = "least_squares_slope_log2_median_real_control_ratio_vs_log2_m"
VERDICT_RETAIN = "RETAIN_STRUCTURED_CIRCUIT_SURPLUS"
VERDICT_REJECT = "REJECT_STRUCTURED_CIRCUIT_SURPLUS"
M12_REQUIRED_WINS = 5
M12_TOTAL_FIXTURES = 6
MEDIAN_RATIO_CEILING = QQ(90) / QQ(100)
REQUIRED_SLOPE_SIGN = "negative"

RAW_RECORD_SCHEMA_VERSION = 1
RAW_RECORD_FIELDS = (
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

CORRECTNESS_AUDITS = (
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


class FrozenSpectrumContract:
    def __init__(self, fixtures, m_values, target_count, control_trials, controls, target_v_mode):
        self.fixtures = tuple(fixtures)
        self.m_values = tuple(m_values)
        self.target_count = ZZ(target_count)
        self.control_trials = ZZ(control_trials)
        self.controls = tuple(controls)
        self.target_v_mode = validate_target_v_mode(target_v_mode)

    def as_dict(self):
        return {
            "fixtures": self.fixtures,
            "m_values": self.m_values,
            "target_count": self.target_count,
            "control_trials": self.control_trials,
            "controls": self.controls,
            "comparators": STRUCTURED_COMPARATORS,
            "target_v_mode": self.target_v_mode,
            "trial_seed_hash": TRIAL_SEED_HASH,
            "raw_record_fields": RAW_RECORD_FIELDS,
            "correctness_audits": CORRECTNESS_AUDITS,
            "verdicts": (VERDICT_RETAIN, VERDICT_REJECT),
        }


def _reject_if_nonpositive(name, value):
    if ZZ(value) <= 0:
        raise ValueError("{} must be positive".format(name))


def _require_exact(name, value, expected):
    if value != expected:
        raise ValueError("{} must equal the frozen structured-spectrum contract".format(name))


def validate_fixtures(fixtures):
    normalized = tuple((ZZ(bits), ZZ(seed), ZZ(order)) for bits, seed, order in fixtures)
    seeds = tuple(seed for _bits, seed, _order in normalized)
    cells = tuple((bits, seed) for bits, seed, _order in normalized)
    if len(set(seeds)) != len(seeds):
        raise ValueError("fixture seeds must be unique")
    if len(set(cells)) != len(cells):
        raise ValueError("fixture cells must be unique")
    return normalized


def validate_controls(controls):
    normalized = tuple(str(control) for control in controls)
    if len(set(normalized)) != len(normalized):
        raise ValueError("controls must be unique")
    unknown = tuple(control for control in normalized if control not in STRUCTURED_CONTROLS)
    if unknown:
        raise ValueError("unknown structured-spectrum control: {}".format(unknown[0]))
    return normalized


def validate_target_v_mode(mode):
    normalized = str(mode)
    if normalized in REJECTED_TARGET_V_MODES:
        raise ValueError("per-relation-v mode is forbidden")
    if normalized != TARGET_V_MODE:
        raise ValueError("unknown target-v mode: {}".format(normalized))
    return normalized


def make_spectrum_contract(
    fixtures=STRUCTURED_FIXTURES,
    m_values=M_VALUES,
    target_count=TARGET_COUNT,
    control_trials=CONTROL_TRIALS,
    controls=STRUCTURED_CONTROLS,
    target_v_mode=TARGET_V_MODE,
):
    validated_fixtures = validate_fixtures(fixtures)
    validated_controls = validate_controls(controls)
    validated_m_values = tuple(ZZ(m) for m in m_values)
    _reject_if_nonpositive("target_count", target_count)
    _reject_if_nonpositive("control_trials", control_trials)
    _require_exact("fixtures", validated_fixtures, STRUCTURED_FIXTURES)
    _require_exact("m_values", validated_m_values, M_VALUES)
    _require_exact("target_count", ZZ(target_count), ZZ(TARGET_COUNT))
    _require_exact("control_trials", ZZ(control_trials), ZZ(CONTROL_TRIALS))
    _require_exact("controls", validated_controls, STRUCTURED_EVALUATED_ORDERS_AND_NULLS)
    _require_exact("target_v_mode", validate_target_v_mode(target_v_mode), TARGET_V_MODE)
    return FrozenSpectrumContract(
        validated_fixtures,
        validated_m_values,
        target_count,
        control_trials,
        validated_controls,
        target_v_mode,
    )


def seed_material(fixture_seed, m, control, trial):
    validate_controls((control,))
    return (
        "{}|fixture_seed={}|m={}|control={}|trial={}".format(
            SEED_DOMAIN, ZZ(fixture_seed), ZZ(m), str(control), ZZ(trial)
        )
    )


def derive_trial_seed(fixture_seed, m, control, trial):
    material = seed_material(fixture_seed, m, control, trial).encode("ascii")
    return ZZ(int(hashlib.sha256(material).hexdigest(), 16))


def exact_median(values):
    ordered = tuple(sorted(QQ(value) for value in values))
    if not ordered:
        raise ValueError("median requires at least one value")
    midpoint = len(ordered) // 2
    if len(ordered) % 2 == 1:
        return ordered[midpoint]
    return (ordered[midpoint - 1] + ordered[midpoint]) / QQ(2)


def censor_relations(relations_to_useful, censored, stream_length):
    if censored:
        return ZZ(stream_length) + 1
    return ZZ(relations_to_useful)


def least_squares_log2_slope(m_values, median_ratios):
    if len(m_values) != len(median_ratios):
        raise ValueError("m_values and median_ratios must have the same length")
    if len(m_values) < 2:
        raise ValueError("slope requires at least two cells")
    xs = tuple(RR(ZZ(m)).log2() for m in m_values)
    ys = tuple(RR(QQ(ratio)).log2() for ratio in median_ratios)
    x_bar = sum(xs) / RR(len(xs))
    y_bar = sum(ys) / RR(len(ys))
    denominator = sum((x - x_bar)^2 for x in xs)
    if denominator == 0:
        raise ValueError("slope requires distinct m values")
    return sum((x - x_bar) * (y - y_bar) for x, y in zip(xs, ys)) / denominator


def gate_thresholds_as_dict():
    return {
        "m12_wins": "5/6",
        "median_ratio_ceiling": "0.90",
        "required_slope": "<0",
    }


STRUCTURED_SPECTRUM_CONTRACT = make_spectrum_contract()
