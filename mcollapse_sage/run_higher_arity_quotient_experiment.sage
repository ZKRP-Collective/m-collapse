import hashlib
import json
import os
from sage.env import SAGE_VERSION

load("mcollapse_sage/rank_guided_mcollapse.sage")
load("mcollapse_sage/symmetry_quotient_mcollapse.sage")
load("mcollapse_sage/higher_arity_quotient.sage")


def json_safe(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    try:
        if value in ZZ:
            return int(value)
    except (TypeError, ValueError):
        pass
    return value


def stream_digest(quotient, anchor_count):
    digest = hashlib.sha256()
    generated = ZZ(0)
    for row in quotient.iter_global_rows(anchor_count):
        payload = [len(row)]
        for anchor, a, b in row:
            encoded_a = int(a + 64)
            encoded_b = int(b + 64)
            if not (0 <= anchor < 256 and 0 <= encoded_a < 256 and 0 <= encoded_b < 256):
                raise ArithmeticError("row exceeds registered digest encoding")
            payload.extend((int(anchor), encoded_a, encoded_b))
        digest.update(bytes(payload))
        generated += 1
    return generated, digest.hexdigest()


def witness_failures(quotient):
    failures = ZZ(0)
    zero = (ZZ(0), ZZ(0))
    for support, classes in quotient.local_classes.items():
        for values, coordinate_witnesses in classes.items():
            if sum(len(items) for items in coordinate_witnesses) != quotient.partial_arity:
                failures += 1
                continue
            expected = (zero,) if support == 0 else values
            for value, witness in zip(expected, coordinate_witnesses):
                total = zero
                for term in witness:
                    if term not in quotient.alphabet:
                        failures += 1
                    total = eisenstein_add(total, term)
                if total != value:
                    failures += 1
    return failures


anchor_count = 12
quotients = {
    "arity-4": HigherArityCoefficientQuotient(3),
    "arity-5": HigherArityCoefficientQuotient(4),
}
catalogs = {}
for label, quotient in quotients.items():
    generated, digest = stream_digest(quotient, anchor_count)
    summary = quotient.summary(anchor_count)
    summary["generated_rows"] = int(generated)
    summary["row_stream_sha256"] = digest
    summary["witness_failures"] = int(witness_failures(quotient))
    summary["generated_count_matches"] = bool(
        generated == quotient.global_count(anchor_count)
    )
    catalogs[label] = summary
    print(
        "%s generated=%s ratio=%s sha256=%s"
        % (
            label,
            generated,
            summary["ordered_to_quotient_ratio"],
            digest,
        )
    )

held_out = ((7, 331), (8, 337), (9, 347), (10, 349), (11, 353), (12, 359))
field_cells = []
for subgroup_bits, seed_value in held_out:
    fixture = find_cm_fixture(subgroup_bits, seed_value)
    cell = {"fixture": fixture.as_dict(), "catalogs": {}}
    for label, quotient in quotients.items():
        generic_count = quotient.global_count(4)
        mapped = {
            map_symbolic_row_to_field(row, fixture)
            for row in quotient.iter_global_rows(4)
        }
        cell["catalogs"][label] = {
            "generic_rows_at_four_anchors": int(generic_count),
            "finite_field_rows": int(len(mapped)),
            "modular_collisions": int(generic_count - len(mapped)),
        }
    field_cells.append(cell)
    print(
        "%s collisions arity4=%s arity5=%s"
        % (
            fixture.label,
            cell["catalogs"]["arity-4"]["modular_collisions"],
            cell["catalogs"]["arity-5"]["modular_collisions"],
        )
    )

passed = (
    catalogs["arity-4"]["local_class_counts"]
    == {0: 1, 1: 18, 2: 168, 3: 288}
    and catalogs["arity-5"]["local_class_counts"]
    == {0: 1, 1: 32, 2: 624, 3: 2880, 4: 3456}
    and all(catalog["generated_count_matches"] for catalog in catalogs.values())
    and all(catalog["witness_failures"] == 0 for catalog in catalogs.values())
    and all(
        item["finite_field_rows"] <= item["generic_rows_at_four_anchors"]
        for cell in field_cells
        for item in cell["catalogs"].values()
    )
)

summary = {
    "experiment": "SageMath higher-arity Eisenstein MCOLLAPSE quotient generation",
    "sage_version": str(SAGE_VERSION),
    "anchor_count": anchor_count,
    "coefficient_ring": "ZZ[lambda]/(lambda^2+lambda+1)",
    "catalogs": catalogs,
    "finite_field_held_out": field_cells,
    "decision": {
        "exact_catalogs_generated": bool(passed),
        "materialize_arity_4_partial_table": False,
        "materialize_arity_5_partial_table": False,
        "next_generator": "balanced pair+pair and pair+pair+term using the 1681-class pair quotient",
    },
}

output_directory = "mcollapse_sage/results"
os.makedirs(output_directory, exist_ok=True)
output_path = os.path.join(output_directory, "higher_arity_quotient_summary.json")
with open(output_path, "w", encoding="utf-8") as output_file:
    json.dump(json_safe(summary), output_file, indent=2, sort_keys=True)
    output_file.write("\n")

print("wrote %s" % output_path)
print("catalog generation: %s" % ("PASS" if passed else "FAIL"))
