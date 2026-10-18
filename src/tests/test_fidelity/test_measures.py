"""Each performance measure on its own, on physical objects (ADR 0004).

Every measure must:
1. report fidelity 1 and loss 0 at the target;
2. return a gradient G with d(loss) = Re<G, dX>, checked against central finite
   differences in the state along complex directions, so sign and conjugation
   slips show up;
3. reject a state of the wrong kind.

The end-to-end check through GRAPE is ``test_solvers/test_grape_grad.py``; this
file localizes a failure to the measure. A new measure is one more ``Case``.
"""
from dataclasses import dataclass
from typing import Callable

import numpy as np
import pytest
from qutip import Qobj, basis, ket2dm, qeye, rand_dm, rand_herm, rand_ket, rand_unitary

from qoc.objectives import StateTransfer
from qoc.performance.base import FunctionalPerformanceMeasure, PerformanceMeasure
from qoc.performance.gate_fidelity import UnitaryAverageGateFidelity
from qoc.performance.state_fidelity import KetStateFidelity
from qoc.performance.hilbert_schmidt import SquaredHSDistance, NormalizedHSOverlap

DIM = 3
EPS = 1e-6


def _inner(a: Qobj, b: Qobj) -> complex:
    """Hilbert-Schmidt inner product tr(a^dag b); a plain overlap for vectors."""
    product = a.dag() @ b
    return complex(product.tr() if isinstance(product, Qobj) else product)


def _complex_ket(rng) -> Qobj:
    return Qobj(rng.normal(size=(DIM, 1)) + 1j * rng.normal(size=(DIM, 1)))


def _complex_operator(rng) -> Qobj:
    return Qobj(rng.normal(size=(DIM, DIM)) + 1j * rng.normal(size=(DIM, DIM)))


def _straight_line(rng, make_direction):
    """Perturb along X + s*dX. Fine for measures defined on the whole space."""
    direction = make_direction(rng)
    return (lambda X, s: X + s * direction), (lambda X: direction)


def _unitary_curve(rng):
    """Perturb along expm(i s H) U, whose tangent at s = 0 is i H U.

    UnitaryAverageGateFidelity validates unitarity, so it is only differentiable along the
    unitary manifold -- which is also the only direction GRAPE ever moves U in.
    """
    H = rand_herm(DIM, seed=rng.integers(2**31))
    return (lambda U, s: (1j * s * H).expm() @ U), (lambda U: 1j * H @ U)


@dataclass
class Case:
    measure: PerformanceMeasure
    make_state: Callable[[int], Qobj]      # seed -> a valid physical object
    make_curve: Callable                   # rng -> (curve(X, s), tangent(X))
    wrong_kind: Qobj
    wrong_kind_error: type[Exception]


def _hs_distance(rho: Qobj, rho_t: Qobj) -> float:
    """||rho - rho_t||_HS**2, a loss: 0 at the target."""
    diff = rho - rho_t
    return float(np.real((diff.dag() @ diff).tr()))


def _hs_distance_gradient(rho: Qobj, rho_t: Qobj) -> Qobj:
    """d||rho - rho_t||**2 = Re<2 (rho - rho_t), drho>."""
    return 2.0 * (rho - rho_t)


def _phase_sensitive_overlap(psi: Qobj, target: Qobj) -> float:
    """Re<t|psi>, a figure of merit: 1 at the target, sensitive to global phase."""
    return float(np.real(complex(target.dag() @ psi)))


def _phase_sensitive_overlap_gradient(psi: Qobj, target: Qobj) -> Qobj:
    """d Re<t|psi> = Re<t, dpsi>: the gradient of the figure of merit, not the loss."""
    return target


CASES = {
    "ket-state-fidelity": Case(
        KetStateFidelity(),
        lambda seed: rand_ket(DIM, seed=seed),
        lambda rng: _straight_line(rng, _complex_ket),
        wrong_kind=rand_dm(DIM, seed=0),
        wrong_kind_error=TypeError,
    ),
    "squared-hs-distance": Case(
        SquaredHSDistance(),
        lambda seed: rand_dm(DIM, seed=seed),  # mixed, not only pure
        lambda rng: _straight_line(rng, _complex_operator),
        wrong_kind=rand_ket(DIM, seed=0),
        wrong_kind_error=TypeError,
    ),
    "normalized-hs-overlap": Case(
        NormalizedHSOverlap(),
        lambda seed: rand_dm(DIM, seed=seed),
        lambda rng: _straight_line(rng, _complex_operator),
        wrong_kind=rand_ket(DIM, seed=0),
        wrong_kind_error=TypeError,
    ),
    "gate": Case(
        UnitaryAverageGateFidelity(),
        lambda seed: rand_unitary(DIM, seed=seed),
        _unitary_curve,
        wrong_kind=rand_ket(DIM, seed=0),
        wrong_kind_error=ValueError,
    ),
    "functional-loss": Case(
        FunctionalPerformanceMeasure.from_loss(
            _hs_distance, gradient=_hs_distance_gradient, accepts="dm"
        ),
        lambda seed: rand_dm(DIM, seed=seed),
        lambda rng: _straight_line(rng, _complex_operator),
        wrong_kind=rand_ket(DIM, seed=0),
        wrong_kind_error=TypeError,
    ),
    "functional-fidelity": Case(
        FunctionalPerformanceMeasure(
            _phase_sensitive_overlap,
            gradient=_phase_sensitive_overlap_gradient,
            accepts="ket",
        ),
        lambda seed: rand_ket(DIM, seed=seed),
        lambda rng: _straight_line(rng, _complex_ket),
        wrong_kind=rand_dm(DIM, seed=0),
        wrong_kind_error=TypeError,
    ),
}


@pytest.fixture(params=list(CASES), ids=list(CASES))
def case(request) -> Case:
    return CASES[request.param]


def test_perfect_at_target(case):
    target = case.make_state(1)
    assert np.isclose(case.measure.fidelity(target, target), 1.0)
    assert np.isclose(case.measure.loss(target, target), 0.0)


@pytest.mark.parametrize("seed", [10, 11, 12])
def test_gradient_matches_finite_differences(case, seed):
    rng = np.random.default_rng(seed)
    target = case.make_state(seed)
    current = case.make_state(seed + 100)
    curve, tangent = case.make_curve(rng)

    G = case.measure.loss_gradient(current, target)
    analytic = np.real(_inner(G, tangent(current)))
    numeric = (
        case.measure.loss(curve(current, EPS), target)
        - case.measure.loss(curve(current, -EPS), target)
    ) / (2 * EPS)

    # Guard against a vacuous pass at a stationary point
    assert abs(numeric) > 1e-3
    assert np.isclose(analytic, numeric, rtol=1e-6, atol=1e-8)


def test_gradient_has_the_physical_shape(case):
    target = case.make_state(1)
    current = case.make_state(2)
    G = case.measure.loss_gradient(current, target)
    assert G.dims == current.dims


def test_rejects_wrong_kind_of_state(case):
    target = case.make_state(1)
    with pytest.raises(case.wrong_kind_error):
        case.measure.loss(case.wrong_kind, target)


def test_declares_what_it_accepts(case):
    assert case.measure.accepts


def test_open_default_prefers_mixed_target_to_its_dominant_eigenstate():
    target = Qobj(np.diag([0.8, 0.2]))
    pure = ket2dm(basis(2, 0))
    objective = StateTransfer(pure, target)

    assert isinstance(objective.performance_measure, SquaredHSDistance)
    assert objective.loss(target) == 0.0
    assert objective.loss(pure) == pytest.approx(0.04)
    # Moving away from the target must increase loss, not improve the score.
    losses = [objective.loss((1 - s) * target + s * pure)
              for s in np.linspace(0, 1, 6)]
    assert np.all(np.diff(losses) > 0)


@pytest.mark.parametrize("dimension", [2, 3, 4])
def test_open_default_distinguishes_states_from_maximally_mixed_target(dimension):
    target = qeye(dimension) / dimension
    pure = ket2dm(basis(dimension, 0))
    objective = StateTransfer(pure, target)

    assert objective.loss(target) == 0.0
    assert objective.loss(pure) == pytest.approx(0.5 * (1 - 1 / dimension))
    assert objective.loss(0.5 * pure + 0.5 * target) < objective.loss(pure)
    assert objective.loss_gradient(target).norm() == 0.0


@pytest.mark.parametrize("seed", [10, 11, 12])
def test_open_score_agrees_with_ket_fidelity_when_both_states_are_pure(seed):
    current, target = rand_ket(DIM, seed=seed), rand_ket(DIM, seed=seed + 100)
    assert SquaredHSDistance().fidelity(ket2dm(current), ket2dm(target)) == pytest.approx(
        KetStateFidelity().fidelity(current, target)
    )


def test_open_score_is_zero_for_orthogonal_pure_states():
    current, target = (ket2dm(basis(2, j)) for j in (0, 1))
    assert SquaredHSDistance().fidelity(current, target) == pytest.approx(0.0)


def test_open_loss_preserves_small_distances_near_target():
    target = qeye(2) / 2
    current = target + 1e-10 * Qobj(np.diag([1., -1.]))
    # Computing loss as 1 - fidelity would round this positive value to zero.
    assert SquaredHSDistance().loss(current, target) == pytest.approx(1e-20, rel=1e-6, abs=0)


def test_normalized_overlap_is_an_explicit_opt_in_to_the_old_objective():
    target = Qobj(np.diag([0.8, 0.2]))
    pure = ket2dm(basis(2, 0))
    objective = StateTransfer(pure, target, performance_measure=NormalizedHSOverlap())
    assert objective.fidelity(pure) == pytest.approx(0.8 / 0.68)
    assert objective.loss(pure) < objective.loss(target)
    # For a pure target this option still reports target population.
    assert NormalizedHSOverlap().fidelity(target, pure) == pytest.approx(0.8)


# FunctionalPerformanceMeasure: what the user-supplied gradient must look like

def test_functional_measure_without_gradient_names_the_fix():
    measure = FunctionalPerformanceMeasure.from_loss(_hs_distance)
    rho = rand_dm(DIM, seed=1)
    with pytest.raises(NotImplementedError, match="gradient="):
        measure.loss_gradient(rho, rho)


def test_functional_gradient_must_accept_current_and_target():
    with pytest.raises(TypeError, match="gradient"):
        FunctionalPerformanceMeasure.from_loss(_hs_distance, gradient=lambda rho: rho)


@pytest.mark.parametrize(
    "bad_gradient, error",
    [
        (lambda rho, rho_t: (rho - rho_t).full(), TypeError),   # not a Qobj
        (lambda rho, rho_t: rand_ket(DIM, seed=0), ValueError),  # not current's shape
        (lambda rho, rho_t: np.nan * rho, ValueError),           # not finite
    ],
    ids=["ndarray", "wrong-dims", "nan"],
)
def test_functional_gradient_output_is_validated(bad_gradient, error):
    measure = FunctionalPerformanceMeasure.from_loss(_hs_distance, gradient=bad_gradient)
    rho = rand_dm(DIM, seed=1)
    with pytest.raises(error):
        measure.loss_gradient(rho, rand_dm(DIM, seed=2))


# FunctionalPerformanceMeasure: declaring what it accepts

def test_functional_measure_checks_nothing_when_accepts_is_undeclared():
    measure = FunctionalPerformanceMeasure.from_loss(lambda x, t: 0.0)
    assert measure.accepts is None
    measure.loss(rand_ket(DIM, seed=1), None)
    measure.loss(rand_dm(DIM, seed=1), None)


def test_functional_measure_may_accept_several_kinds():
    measure = FunctionalPerformanceMeasure.from_loss(
        _hs_distance, gradient=_hs_distance_gradient, accepts=("ket", "dm")
    )
    assert measure.accepts == {"ket", "dm"}
    for state in (rand_ket(DIM, seed=1), rand_dm(DIM, seed=1)):
        measure.loss_gradient(state, state)


def test_functional_measure_checks_kind_before_calling_the_gradient():
    def gradient(rho, rho_t):
        raise AssertionError("gradient called on the wrong kind of state")

    measure = FunctionalPerformanceMeasure.from_loss(
        _hs_distance, gradient=gradient, accepts="dm"
    )
    with pytest.raises(TypeError, match="density matrix"):
        measure.loss_gradient(rand_ket(DIM, seed=1), rand_dm(DIM, seed=1))


@pytest.mark.parametrize("accepts", ["density matrix", (), ["ket", "oper"]])
def test_functional_measure_rejects_unknown_kinds(accepts):
    with pytest.raises(ValueError, match="accepts"):
        FunctionalPerformanceMeasure(_phase_sensitive_overlap, accepts=accepts)
