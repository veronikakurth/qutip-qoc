import pytest
import numpy as np
from qutip import sigmax, sigmay, sigmaz

from qoc.algorithms.grape import _step_propagators
from qoc.systems.closed import ClosedSystem


@pytest.fixture
def zero_system():
    """H0 = 0, one x-control. Produces identity when u = 0."""
    return ClosedSystem(H0=0 * sigmax(), H_controls=[0.5 * sigmax()])


@pytest.fixture
def x_system():
    return ClosedSystem(H0=0 * sigmax(), H_controls=[0.5 * sigmax()])


def compute_single(system, u, dt):
    """Single slice, single control."""
    return _step_propagators(system, np.array([[u]]), dt)[0]


class TestStepPropagators:

    def test_zero_amplitude_gives_identity(self, zero_system):
        result = compute_single(zero_system, u=0.0, dt=0.5)
        assert np.allclose(result, np.eye(2), atol=1e-12)

    def test_unitarity(self, x_system):
        result = compute_single(x_system, u=1.0, dt=0.5)
        assert np.allclose(result @ result.conj().T, np.eye(2), atol=1e-12)

    def test_half_rotation(self, x_system):
        # θ = u·dt/2 = π/4  →  U = (1/√2)(I − i·σₓ)
        dt = 0.5
        u = np.pi / (2 * dt)
        s = 1 / np.sqrt(2)

        result = compute_single(x_system, u=u, dt=dt)
        expected = np.array([[s, -1j * s],
                             [-1j * s, s]])

        assert np.allclose(result, expected, atol=1e-12)

    def test_composition(self, x_system):
        # U(u, 2dt) = U(u, dt)²
        dt = 0.3
        u = 0.7
        amps = np.array([[u]])

        single = _step_propagators(x_system, amps, dt)[0]
        double = _step_propagators(x_system, amps, 2 * dt)[0]

        assert np.allclose(double, single @ single, atol=1e-14)
