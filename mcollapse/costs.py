"""Primitive operation accounting for registered affine formulas.

The affine addition vector assumes one inversion, two multiplications, and one
squaring. Affine doubling assumes one inversion, three multiplications, and
two squarings. ``PHI`` is one field multiplication, ``PROBE`` records one
table operation, and ``NORMALIZE`` records one coefficient reduction.
Cancellation requires no tracked field arithmetic but still records one
point-add branch; identity is a true no-op. The operation vectors remain
available even when a caller also supplies calibration weights.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from math import isfinite
from numbers import Real
from typing import Mapping


@dataclass(frozen=True)
class CostVector:
    fp_mul: int = 0
    fp_square: int = 0
    fp_inv: int = 0
    point_add: int = 0
    point_double: int = 0
    endomorphism: int = 0
    table_probe: int = 0
    normalization: int = 0

    def __post_init__(self) -> None:
        for field in fields(self):
            value = getattr(self, field.name)
            if type(value) is not int:
                raise TypeError(f"{field.name} must be an integer")
            if value < 0:
                raise ValueError(f"{field.name} must be nonnegative")

    def __add__(self, other: object) -> CostVector:
        if not isinstance(other, CostVector):
            return NotImplemented
        return CostVector(
            *(getattr(self, field.name) + getattr(other, field.name) for field in fields(self))
        )

    def __mul__(self, factor: object) -> CostVector:
        if type(factor) is not int:
            raise TypeError("cost scaling factor must be an integer")
        if factor < 0:
            raise ValueError("cost scaling factor must be nonnegative")
        return CostVector(*(getattr(self, field.name) * factor for field in fields(self)))

    def __rmul__(self, factor: object) -> CostVector:
        return self * factor

    def calibrated(self, weights: Mapping[str, Real]) -> float:
        if not isinstance(weights, Mapping):
            raise TypeError("weights must be a mapping")
        names = tuple(field.name for field in fields(self))
        if set(weights) != set(names):
            raise ValueError("weights must contain exactly every cost field")
        total = 0.0
        for name in names:
            weight = weights[name]
            if isinstance(weight, bool) or not isinstance(weight, Real):
                raise TypeError("calibration weights must be numeric")
            if not isfinite(weight) or weight < 0:
                raise ValueError("calibration weights must be finite and nonnegative")
            total += getattr(self, name) * weight
        return total


AFFINE_ADD = CostVector(fp_mul=2, fp_square=1, fp_inv=1, point_add=1)
AFFINE_DOUBLE = CostVector(fp_mul=3, fp_square=2, fp_inv=1, point_double=1)
PHI = CostVector(fp_mul=1, endomorphism=1)
PROBE = CostVector(table_probe=1)
NORMALIZE = CostVector(normalization=1)
CANCELLATION = CostVector(point_add=1)
IDENTITY_NOOP = CostVector()
