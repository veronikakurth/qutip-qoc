"""Hilbert-Schmidt distance and overlap measures on density matrices.

Both declare their physical domain via ``accepts`` and work independently of
system dynamics or numerical encoding. Neither is a FidelityMeasure.
"""
from dataclasses import dataclass

import numpy as np
from qutip import Qobj

from .base import PerformanceMeasure


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


class SquaredHSDistance(PerformanceMeasure):
    """Squared Hilbert-Schmidt distance loss for density matrices, scaled by 1/2:

        loss = 0.5 * ||rho - rho_t||_HS**2
        F = 1 - loss

    The loss is nonnegative and vanishes only at the target, including mixed
    targets. For normalized density matrices F lies in [0, 1]; when both
    states are pure it equals their squared ket overlap. For mixed states
    this is an HS similarity score, not the Uhlmann fidelity or the target
    population. The fixed factor of one half preserves the normalization
    introduced for density-matrix matching; there is no target-dependent scale.
    """

    accepts = frozenset({"dm"})

    def loss(self, current: Qobj, target: Qobj) -> float:
        """Half the squared HS distance, evaluated directly near the target."""
        self._check_kind(current)
        self._check_kind(target)
        diff = current - target
        return 0.5 * float(np.real((diff.dag() @ diff).tr()))

    def loss_gradient(self, current: Qobj, target: Qobj) -> Qobj:
        """d(loss) = Re<rho - rho_t, drho>, so G = rho - rho_t."""
        self._check_kind(current)
        self._check_kind(target)
        return current - target

    def fidelity_formula(self):
        """Display the HS similarity score in plain text or LaTeX."""
        return Formula(
            text="F = 1 - 1/2 ||ρ - ρ_tar||_HS²",
            latex=(
                r"F = 1 - \frac{1}{2}"
                r"\lVert \rho - \rho_{\mathrm{tar}} \rVert_{\mathrm{HS}}^2"
            ),
        )


class NormalizedHSOverlap(PerformanceMeasure):
    """Explicit normalized overlap objective for density matrices.

    F = Re tr(rho_t^dag rho) / tr(rho_t^dag rho_t), with loss = 1 - F.
    For a pure target, F is its population. For mixed targets F can exceed 1
    and the target need not minimize the loss; for a maximally mixed target
    the loss is constant.
    Choose SquaredHSDistance for matching arbitrary density matrices.
    """

    accepts = frozenset({"dm"})

    def loss(self, current: Qobj, target: Qobj) -> float:
        return 1.0 - self.fidelity(current, target)

    def fidelity(self, current: Qobj, target: Qobj) -> float:
        """Normalized real-valued Hilbert-Schmidt overlap"""
        self._check_kind(current)
        overlap = (target.dag() @ current).tr()
        return float(np.real(overlap) / _target_norm_sq(target))

    def loss_gradient(self, current: Qobj, target: Qobj) -> Qobj:
        """loss = 1 - Re tr(rho_t^dag rho) / s is linear in rho, so
        d(loss) = Re< -rho_t/s, drho>.
        """
        self._check_kind(current)
        return -target / _target_norm_sq(target)

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


def _target_norm_sq(target: Qobj) -> float:
    """Tr(rho_t^dag rho_t) = ||rho_t||_HS**2, the normalization that puts F = 1 at rho = rho_t.
    """
    norm_sq = float(np.real((target.dag() @ target).tr()))
    if norm_sq == 0.0:
        raise ValueError("target state has zero norm; fidelity is undefined")
    return norm_sq
