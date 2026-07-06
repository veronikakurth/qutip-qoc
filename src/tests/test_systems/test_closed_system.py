"""
Unit tests for ClosedSystem.
"""
import numpy as np
import pytest
from scipy.linalg import expm
from qutip import sigmax, sigmaz, Qobj

from qoc.systems.closed import ClosedSystem
from qoc.algorithms.grape import _step_propagators


@pytest.fixture
def system():
    return ClosedSystem(H0=sigmaz() / 2, H_controls=[sigmax() / 2])


@pytest.fixture
def times():
    return np.linspace(0, 1, 5)


@pytest.fixture
def pi_pulse(system, times):
    N = len(times)
    dt = times[1] - times[0]
    return np.array([[np.pi / dt / system.controls[0].norm()] * N])


@pytest.fixture
def zero_pulse(system, times):
    return np.zeros((system.n_controls, len(times)))


class TestBuildHamiltonian:

    def test_returns_list_with_drift_and_controls(self, system, times, zero_pulse):
        H = system.build_generator(zero_pulse)
        assert isinstance(H, list)
        assert H[0] == system.drift
        assert len(H) == 1 + system.n_controls

    def test_control_coefficient_shape(self, system, times, pi_pulse):
        H = system.build_generator(pi_pulse)
        _, coeff = H[1]
        assert len(coeff) == len(times)


class TestBuildHamiltonianTimeJ:

    def test_zero_controls_returns_drift(self, system, zero_pulse):
        H_j = system.build_generator_time_j(zero_pulse, j=0)
        assert H_j == system.drift

    def test_returns_qobj(self, system, pi_pulse):
        H_j = system.build_generator_time_j(pi_pulse, j=0)
        assert isinstance(H_j, Qobj)

    def test_control_contribution(self, system, times):
        u = np.array([[2.0] * len(times)])
        H_j = system.build_generator_time_j(u, j=0)
        expected = system.drift + 2.0 * system.controls[0]
        assert H_j == expected


class TestStepPropagators:

    def test_returns_n_propagators(self, system, times, pi_pulse):
        Us = _step_propagators(system, pi_pulse, dt=times[1] - times[0])
        assert len(Us) == len(times)

    def test_each_propagator_is_unitary(self, system, times, pi_pulse):
        Us = _step_propagators(system, pi_pulse, dt=times[1] - times[0])
        for j, U in enumerate(Us):
            residual = np.linalg.norm(U @ U.conj().T - np.eye(2))
            assert residual < 1e-12, f"Propagator {j} is not unitary"

    def test_zero_pulse_gives_drift_only(self, system, times, zero_pulse):
        dt = times[1] - times[0]
        Us = _step_propagators(system, zero_pulse, dt=dt)
        expected = expm(-1j * system.drift.full() * dt)
        for j, U in enumerate(Us):
            assert np.allclose(U, expected, atol=1e-12), f"Propagator {j} differs from drift-only"
