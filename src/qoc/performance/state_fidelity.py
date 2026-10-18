"""Ket fidelity and construction-time selection of default state measures.

Selection uses physical state kind, never the system or its encoding.
Density-matrix distance and overlap formulas live in hilbert_schmidt.py.
"""
from typing import Literal

import numpy as np
from qutip import Qobj

from .base import FidelityMeasure, PerformanceMeasure
from .hilbert_schmidt import SquaredHSDistance


class KetStateFidelity(FidelityMeasure):
    """Squared overlap fidelity between normalized kets:

        F = |<t|psi>|**2
    """

    accepts = frozenset({"ket"})

    # TODO: decide how to enable phase-sensitive fidelity
    def fidelity(self, current: Qobj, target: Qobj) -> float:
        self._check_kind(current)
        return float(np.abs(complex(target.dag() @ current)) ** 2)

    def loss_gradient(self, current: Qobj, target: Qobj) -> Qobj:
        """loss = 1 - |c|**2 with c = <t|psi>, so
        d(loss) = -2 Re(conj(c) <t|dpsi>) = Re< -2 c t, dpsi>.
        """
        self._check_kind(current)
        c = complex(target.dag() @ current)
        return -2.0 * c * target


def state_measure_for(state_type: Literal["ket", "dm"]) -> PerformanceMeasure:
    """Select a default formula once from the physical state kind."""
    if state_type == "ket":
        return KetStateFidelity()
    if state_type == "dm":
        return SquaredHSDistance()
    raise ValueError(
        f"Unknown state_type {state_type!r}; expected 'ket' or 'dm'"
    )
