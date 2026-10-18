"""Objectives reject semantic domain mismatches before evaluating a measure."""
import pytest
from qutip import basis, ket2dm, qeye, sigmax

from qoc import (
    FunctionalPerformanceMeasure,
    GateSynthesis,
    KetStateFidelity,
    NormalizedHSOverlap,
    SquaredHSDistance,
    StateTransfer,
    UnitaryAverageGateFidelity,
)


def _objective(kind, measure):
    if kind == "ket":
        return StateTransfer(basis(2, 0), basis(2, 1), measure)
    if kind == "dm":
        return StateTransfer(ket2dm(basis(2, 0)), qeye(2) / 2, measure)
    return GateSynthesis(qeye(2), sigmax(), measure)


@pytest.mark.parametrize("kind", ["ket", "dm", "unitary"])
@pytest.mark.parametrize("measure_type, domain", [
    (KetStateFidelity, "ket"),
    (SquaredHSDistance, "dm"),
    (NormalizedHSOverlap, "dm"),
    (UnitaryAverageGateFidelity, "unitary"),
])
def test_builtin_domains_are_checked_at_objective_construction(kind, measure_type, domain):
    measure = measure_type()
    if kind == domain:
        assert _objective(kind, measure).performance_measure is measure
    else:
        with pytest.raises(TypeError) as exc:
            _objective(kind, measure)
        assert str(exc.value) == (
            f"{measure_type.__name__} accepts [{domain}], "
            f"but the objective uses {kind!r}."
        )


def _must_not_evaluate(*args):
    raise AssertionError("Objective construction evaluated the measure")


@pytest.mark.parametrize("kind", ["ket", "dm", "unitary"])
@pytest.mark.parametrize("accepts", [None, ("ket", "dm"), ("unitary",)])
def test_custom_domains_are_checked_without_evaluating_functions(kind, accepts):
    measure = FunctionalPerformanceMeasure.from_loss(
        _must_not_evaluate, gradient=_must_not_evaluate, accepts=accepts,
    )
    if accepts is None or kind in accepts:
        assert _objective(kind, measure).performance_measure is measure
    else:
        with pytest.raises(TypeError, match="FunctionalPerformanceMeasure accepts"):
            _objective(kind, measure)


@pytest.mark.parametrize("kind", ["ket", "dm", "unitary"])
def test_measure_type_is_validated_before_domain_check(kind):
    with pytest.raises(TypeError, match="PerformanceMeasure"):
        _objective(kind, object())


def test_states_are_validated_before_measure_domain():
    with pytest.raises(ValueError, match="same type"):
        StateTransfer(basis(2, 0), ket2dm(basis(2, 1)), SquaredHSDistance())


def test_gate_dimensions_are_validated_before_measure_domain():
    with pytest.raises(ValueError, match="dimensions do not match"):
        GateSynthesis(qeye(2), qeye(3), SquaredHSDistance())
