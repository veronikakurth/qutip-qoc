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
from qutip import Qobj, rand_dm, rand_herm, rand_ket, rand_unitary

from qoc.performance.base import PerformanceMeasure
from qoc.performance.gate_fidelity import GateFidelity
from qoc.performance.state_fidelity import ClosedStateFidelity, OpenStateFidelity

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

    GateFidelity validates unitarity, so it is only differentiable along the
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


CASES = {
    "closed-state": Case(
        ClosedStateFidelity(),
        lambda seed: rand_ket(DIM, seed=seed),
        lambda rng: _straight_line(rng, _complex_ket),
        wrong_kind=rand_dm(DIM, seed=0),
        wrong_kind_error=TypeError,
    ),
    "open-state": Case(
        OpenStateFidelity(),
        lambda seed: rand_dm(DIM, seed=seed),  # mixed, not only pure
        lambda rng: _straight_line(rng, _complex_operator),
        wrong_kind=rand_ket(DIM, seed=0),
        wrong_kind_error=TypeError,
    ),
    "gate": Case(
        GateFidelity(),
        lambda seed: rand_unitary(DIM, seed=seed),
        _unitary_curve,
        wrong_kind=rand_ket(DIM, seed=0),
        wrong_kind_error=ValueError,
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
