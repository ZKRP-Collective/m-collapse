"""Tests for the exact synthetic Circuit-MCOLLAPSE scaling oracle."""

load("mcollapse_sage/circuit_mcollapse.sage")
load("mcollapse_sage/circuit_scaling_gate.sage")


for width in (4, 8, 12):
    cell = run_circuit_scaling_cell(width, 8, 29001 + width)
    if cell["solved"] != cell["trials"]:
        raise AssertionError("synthetic Circuit trial did not terminate")
    if cell["exact_recoveries"] != cell["trials"]:
        raise AssertionError("synthetic Circuit trial recovered a wrong scalar")
print("Circuit-MCOLLAPSE scaling tests: PASS")

