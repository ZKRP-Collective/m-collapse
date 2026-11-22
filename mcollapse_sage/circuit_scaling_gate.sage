"""Synthetic exact-relation scaling gate for Circuit-MCOLLAPSE over secp256k1 n."""


CLASSIC_SECP256K1_N = ZZ(
    0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141
)
CLASSIC_SECP256K1_LAMBDA = ZZ(
    0x5363AD4CC05C30E0A5261C028812645A122E22EA20816678DF02967C1B23BD72
)


def _random_nonzero(field):
    value = field(0)
    while value == 0:
        value = field(ZZ.random_element(1, field.cardinality()))
    return value


def _random_support(width, support_size=3):
    support = set()
    while len(support) < support_size:
        support.add(ZZ.random_element(width))
    return tuple(sorted(support))


def run_circuit_scaling_trial(width, seed_value):
    width = ZZ(width)
    if width < 3:
        raise ValueError("circuit scaling width must be at least three")
    set_random_seed(ZZ(seed_value))
    field = GF(CLASSIC_SECP256K1_N)
    eigenvalue = field(CLASSIC_SECP256K1_LAMBDA)
    if eigenvalue^2 + eigenvalue + 1 != 0:
        raise ArithmeticError("secp256k1 GLV eigenvalue check failed")
    alphabet = tuple(
        field(sign * weight) * eigenvalue^exponent
        for sign in (ZZ(1), ZZ(-1))
        for weight in (ZZ(1), ZZ(2))
        for exponent in range(3)
    )
    hidden = _random_nonzero(field)
    anchor_logs = tuple(_random_nonzero(field) for _index in range(width))
    basis = OnlineAnchorCircuitBasis(field, width)
    ceiling = 2 * width + 16
    for relation in range(1, ceiling + 1):
        row = {
            anchor: alphabet[ZZ.random_element(len(alphabet))]
            for anchor in _random_support(width)
        }
        v = _random_nonzero(field)
        rhs = sum(
            coefficient * anchor_logs[anchor]
            for anchor, coefficient in row.items()
        ) - v * hidden
        residual, q_coefficient, reduced_rhs, _operations = basis.reduce(
            row, -v, rhs
        )
        if residual:
            basis.commit(residual, q_coefficient, reduced_rhs)
            continue
        if q_coefficient == 0:
            if reduced_rhs != 0:
                raise ArithmeticError("synthetic exact system became inconsistent")
            continue
        recovered = reduced_rhs / q_coefficient
        return {
            "solved": True,
            "relations_processed": ZZ(relation),
            "anchor_rank": basis.rank,
            "recovered_exactly": bool(recovered == hidden),
        }
    return {
        "solved": False,
        "relations_processed": ceiling,
        "anchor_rank": basis.rank,
        "recovered_exactly": False,
    }


def _exact_median(values):
    ordered = sorted(values)
    midpoint = len(ordered) // 2
    if len(ordered) % 2:
        return QQ(ordered[midpoint])
    return QQ(ordered[midpoint - 1] + ordered[midpoint]) / 2


def run_circuit_scaling_cell(width, trials, seed_value):
    results = tuple(
        run_circuit_scaling_trial(width, ZZ(seed_value) + 1009 * trial)
        for trial in range(ZZ(trials))
    )
    relations = [result["relations_processed"] for result in results]
    ranks = [result["anchor_rank"] for result in results]
    median_relations = _exact_median(relations)
    median_rank = _exact_median(ranks)
    return {
        "width": ZZ(width),
        "trials": ZZ(trials),
        "solved": ZZ(sum(result["solved"] for result in results)),
        "exact_recoveries": ZZ(sum(result["recovered_exactly"] for result in results)),
        "median_relations": median_relations,
        "median_anchor_rank": median_rank,
        "median_relations_over_m": median_relations / width,
        "median_rank_over_m": median_rank / width,
        "minimum_relations": min(relations),
        "maximum_relations": max(relations),
    }


def circuit_scaling_log_slope(cells):
    real = RealField(128)
    xs = [log(real(cell["width"]), 2) for cell in cells]
    ys = [log(real(cell["median_relations"]), 2) for cell in cells]
    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    numerator = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    denominator = sum((x - mean_x)^2 for x in xs)
    return float(numerator / denominator)

