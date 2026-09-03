"""
State fidelity, one class per state encoding.

The closed and open forms are different functionals:
quadratic in the ket, linear in the density matrix.

Which one applies is decided by the *encoding*, which is ``System``'s knowledge.
So the choice is made once, when the objective is built and the state type is known.
"""

from abc import abstractmethod

import numpy as np
from qutip import Qobj

from qoc.systems.base import StateType

from .base import PerformanceMeasure

from dataclasses import dataclass

@dataclass
class Formula:
    text: str
    latex: str

    def __str__(self) -> str:
        return self.text

    def __repr__(self) -> str:
        return self.text

    def _repr_latex_(self) -> str:
        return f"$$ {self.latex} $$"

# For state transfer: Basically, we now have one class per formula per state representation
class StateFidelity(PerformanceMeasure):
    """Base for state-fidelity measures: ``loss = 1 - fidelity``.

    Not instantiable. Use :func:`state_fidelity_for`, or let
    ``StateTransfer`` pick the variant matching its states.
    """

    #: Encoding this measure's inputs must arrive in, as a Qobj ``type``.
    _expected_type: str

    def loss(self, current: Qobj, target: Qobj) -> float:
        return 1.0 - self.fidelity(current, target)

    @abstractmethod
    def fidelity(self, current: Qobj, target: Qobj) -> float:
        """Figure of merit; 1 = perfect match."""

    @abstractmethod
    def loss_gradient(self, current: Qobj, target: Qobj) -> Qobj:
        """d(loss)/d(current)."""

    def _check_encoding(self, current: Qobj) -> None:
        """Validate if a measure is paired with the correct kind of system."""
        if current.type != self._expected_type:
            raise TypeError(
                f"{type(self).__name__} expects states encoded as "
                f"{self._expected_type!r}, got {current.type!r}. This usually "
                f"means the objective's performance measure does not match "
                f"the system it was paired with."
            )


class ClosedStateFidelity(StateFidelity):
    """Fidelity between kets (closed systems):

        F = |<t|psi>|**2
    """

    _expected_type = "ket"
    
    # TODO: decide how to enable phase-sensitive fidelity
    def fidelity(self, current: Qobj, target: Qobj) -> float:
        self._check_encoding(current)
        return float(np.abs(complex(target.dag() @ current)) ** 2)

    def loss_gradient(self, current: Qobj, target: Qobj) -> Qobj:
        """loss = 1 - |c|**2 with c = <t|psi>, so
        d(loss) = -2 Re(conj(c) <t|dpsi>) = Re< -2 c t, dpsi>.
        """
        self._check_encoding(current)
        c = complex(target.dag() @ current)
        return -2.0 * c * target


class OpenStateFidelity(StateFidelity):
    """Fidelity between vectorized density matrices (open systems):

        F = Re tr(rho_t^dag rho) / tr(rho_t**2)
    """

    _expected_type = "operator-ket"
 
    def fidelity(self, current: Qobj, target: Qobj) -> float:
        """Normalized Hilbert-Schmidt superoperator overlap"""
        self._check_encoding(current)
        overlap = target.dag() @ current
        return float(np.real(overlap) / float(_target_norm(target)))

    # In GRAPE, used for the final costate
    # Note: in the case of open state transfer with Hilbert-Schmidt superoperator overlap as fidelity,
    # ``current`` is not used for the gradient of loss. 
    # But we have to keep to one signature for ``PerformanceMeasure.loss_gradient`` since the exact formula
    # may vary depending on ``PerformanceMeasure.fidelity``
    def loss_gradient(self, current: Qobj, target: Qobj) -> Qobj:
        """loss = 1 - Re<<t|rho>> / s is linear in rho, so
        d(loss) = Re< -t/s, drho>.
        """
        self._check_encoding(current)
        return -target / float(_target_norm(target))

    def fidelity_formula(self):
        """Gives Latex representation, but will be formatted as text when Latex rendering is not possible."""
        hs_overlap = Formula(
        text=(
            "F = Re⟨⟨ρ_tar | ρ⟩⟩ / ||ρ_tar||_HS²\n"
            "  = Re Tr(ρ_tar† ρ) / Tr(ρ_tar† ρ_tar)"
        ),
        latex=(
            r"F = \frac{"
            r"\operatorname{Re}\langle\!\langle \rho_{\mathrm{tar}} \mid \rho \rangle\!\rangle"
            r"}{"
            r"\lVert \rho_{\mathrm{tar}} \rVert_{\mathrm{HS}}^2"
            r"}"
            r"="
            r"\frac{"
            r"\operatorname{Re}\,\mathrm{Tr}(\rho_{\mathrm{tar}}^\dagger \rho)"
            r"}{"
            r"\mathrm{Tr}(\rho_{\mathrm{tar}}^\dagger \rho_{\mathrm{tar}})"
            r"}"
            )
        )
        return hs_overlap


def state_fidelity_for(state_type: StateType) -> StateFidelity:
    """The state-fidelity variant for a ``"ket"`` or ``"dm"`` system."""
    if state_type == "ket":
        return ClosedStateFidelity()
    if state_type == "dm":
        return OpenStateFidelity()
    raise ValueError(
        f"Unknown state_type {state_type!r}; expected 'ket' or 'dm'"
    )

# todo: do we really want a square here?
def _target_norm(target: Qobj) -> float:
    norm_sq = float(target.norm()) #** 2
    if norm_sq == 0.0:
        raise ValueError("target state has zero norm; fidelity is undefined")
    return norm_sq
