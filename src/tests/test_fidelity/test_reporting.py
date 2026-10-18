"""Reporting is optional and never defines the optimized loss."""
import numpy as np
import pytest
from qutip import basis, ket2dm, sigmax
from scipy.optimize import OptimizeResult

from qoc.algorithms.grape import GRAPE
from qoc.objectives import StateTransfer
from qoc.performance import (
    FunctionalPerformanceMeasure, KetStateFidelity, NormalizedHSOverlap,
    PerformanceMeasure, SquaredHSDistance,
)
from qoc.problem import OptimalControlProblem
from qoc.pulse import PiecewiseConstant
from qoc.systems.controlled_system import ControlledSystem


def test_only_explicit_fidelities_have_fidelity():
    ket = basis(2, 0)
    rho = ket2dm(ket)
    assert KetStateFidelity().report(ket, ket) == {"fidelity": 1.0}
    assert SquaredHSDistance().report(rho, rho) == {"hs_distance_squared": 0.0}
    assert NormalizedHSOverlap().report(rho, rho) == {"normalized_hs_overlap": 1.0}
    for measure in [SquaredHSDistance(), NormalizedHSOverlap()]:
        assert not hasattr(measure, "fidelity")
        with pytest.raises(TypeError, match="report"):
            StateTransfer(rho, rho, measure).fidelity(rho)


@pytest.mark.parametrize("factory", [FunctionalPerformanceMeasure, FunctionalPerformanceMeasure.from_loss])
def test_functional_reporting_is_explicit(factory):
    ket = basis(2, 0)
    assert factory(lambda x, t: 12.0).report(ket, ket) == {}
    measure = factory(lambda x, t: 12.0, report=lambda x, t: {"population": 0.25})
    assert measure.report(ket, ket) == {"population": 0.25}
    assert measure.loss(ket, ket) == (12.0 if factory == FunctionalPerformanceMeasure.from_loss else -11.0)


@pytest.mark.parametrize("value, error", [({"x": np.nan}, ValueError), ({"x": np.complex128(1 + 2j)}, TypeError), ({"x": [1]}, TypeError), ({1: 0.0}, TypeError), (0.5, TypeError)])
def test_functional_report_validation(value, error):
    measure = FunctionalPerformanceMeasure.from_loss(lambda x, t: 7, report=lambda x, t: value)
    with pytest.raises(error):
        measure.report(basis(2, 0), basis(2, 0))


@pytest.mark.parametrize("with_report", [False, True])
def test_grape_loss_only_measure_and_independent_reporting(with_report):
    reports = []

    class CustomLoss(PerformanceMeasure):
        def loss(self, current, target):
            return 7.0 + 3.0 * (current - target).norm() ** 2

        def loss_gradient(self, current, target):
            return 6.0 * (current - target)

        if with_report:
            def report(self, current, target):
                reports.append(current)
                return {"diagnostic": -42.0}

    class TrialOptimizer:
        def minimize(self, fun, x0, callback, **kwargs):
            # Evaluate a rejected trial last, forcing history to recover the
            # accepted point rather than recording the latest trial loss.
            accepted = x0 + 0.1
            fun(accepted)
            fun(x0 + 0.8)
            callback(accepted)
            return OptimizeResult(x=accepted, nit=1, success=True)

    objective = StateTransfer(basis(2, 0), basis(2, 1), CustomLoss())
    system = ControlledSystem.closed(H0=0 * sigmax(), H_controls=[sigmax()])
    problem = OptimalControlProblem(system, objective)
    parameterization = PiecewiseConstant(1, 1.0, 4)
    algorithm = GRAPE(parameterization, optimizer=TrialOptimizer())
    theta = np.full(4, 0.2)
    evaluations = []
    fun = algorithm.build_loss_and_grad(problem, loss_history=evaluations)
    initial_loss = fun(theta)[0]
    accepted_loss = fun(theta + 0.1)[0]
    assert evaluations == [initial_loss, accepted_loss]
    assert reports == []
    result = algorithm.solve(problem, theta.reshape(1, 4))
    assert result.loss > 1.0
    assert result.loss == pytest.approx(objective.loss(result.final_state))
    assert result.loss_history == pytest.approx([initial_loss, accepted_loss])
    assert result.fidelity is None
    assert result.metrics == ({"diagnostic": -42.0} if with_report else {})
    assert len(reports) == int(with_report)
