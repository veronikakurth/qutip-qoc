"""Public measure construction, extension and import contracts."""
import pytest
from qutip import basis, ket2dm, qeye, sigmax

import qoc
import qoc.performance as performance
from qoc import (
    FidelityMeasure,
    GateSynthesis,
    KetStateFidelity,
    NormalizedHSOverlap,
    PerformanceMeasure,
    SquaredHSDistance,
    StateTransfer,
    UnitaryAverageGateFidelity,
    state_measure_for,
)
import qoc.performance.gate_fidelity as gate_module
import qoc.performance.state_fidelity as state_module


@pytest.mark.parametrize("base", [PerformanceMeasure, FidelityMeasure])
def test_base_requires_its_primitive(base):
    with pytest.raises(TypeError, match="abstract"):
        base()


def test_custom_fidelity_needs_no_gradient_for_loss_evaluation():
    class ConstantFidelity(FidelityMeasure):
        accepts = frozenset({"ket"})

        def fidelity(self, current, target):
            return 0.25

    objective = StateTransfer(basis(2, 0), basis(2, 1), ConstantFidelity())
    assert objective.fidelity(basis(2, 0)) == 0.25
    assert objective.loss(basis(2, 0)) == 0.75
    with pytest.raises(NotImplementedError, match="ConstantFidelity.*loss_gradient"):
        objective.loss_gradient(basis(2, 0))


def test_default_measures_follow_physical_domains():
    ket = StateTransfer(basis(2, 0), basis(2, 1)).performance_measure
    dm = StateTransfer(ket2dm(basis(2, 0)), qeye(2) / 2).performance_measure
    gate = GateSynthesis(qeye(2), sigmax()).performance_measure

    assert type(ket) is KetStateFidelity
    assert type(dm) is SquaredHSDistance
    assert type(gate) is UnitaryAverageGateFidelity
    assert ket.accepts == {"ket"}
    assert dm.accepts == {"dm"}
    assert gate.accepts == {"unitary"}
    assert isinstance(ket, FidelityMeasure)
    assert isinstance(gate, FidelityMeasure)
    assert isinstance(dm, PerformanceMeasure)
    assert not isinstance(dm, FidelityMeasure)
    assert not isinstance(NormalizedHSOverlap(), FidelityMeasure)


def test_state_measure_factory_selects_formulas_by_domain():
    for kind, cls in [("ket", KetStateFidelity), ("dm", SquaredHSDistance)]:
        assert type(state_measure_for(kind)) is cls
    with pytest.raises(ValueError, match="state_type"):
        state_measure_for("unitary")


def test_public_packages_export_the_measure_hierarchy():
    for cls in [
        PerformanceMeasure, FidelityMeasure, KetStateFidelity,
        SquaredHSDistance, UnitaryAverageGateFidelity, NormalizedHSOverlap,
        performance.FunctionalPerformanceMeasure,
    ]:
        assert getattr(qoc, cls.__name__) is cls
        assert getattr(performance, cls.__name__) is cls


def test_legacy_measure_names_are_not_exposed():
    for module in [qoc, performance, state_module, gate_module]:
        for name in ["StateFidelity", "ClosedStateFidelity", "OpenStateFidelity",
                     "GateFidelity", "state_fidelity_for"]:
            assert not hasattr(module, name)
