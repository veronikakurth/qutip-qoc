import pytest
import numpy as np
from qutip import sigmax, sigmay, sigmaz

from qoc.dynamics.propagator import StepPropagator
from qoc.systems.closed import ClosedSystem

# --- fixtures: object construction only ---

@pytest.fixture
def propagator():
    return StepPropagator()

@pytest.fixture
def zero_system():
    """H0 = 0, one x-control. Produce identity when u = 0."""
    return ClosedSystem(H0=0 * sigmax(), H_controls=[0.5 * sigmax()])

@pytest.fixture
def x_system():
    return ClosedSystem(H0=0*sigmax(), H_controls=[0.5*sigmax()])

# ---- helpers ----

def compute_single(propagator, system, u, dt):
    """Single slice and single control"""
    return propagator.compute(system, N=1, control_amplitudes=np.array([[u]]), dt=dt)[0]

# ---- tests ---
# TODO: once StepPropagator contains propagation function for a single slice, refactor tests - don't use indexing of the lsit
class TestStepPropagatorCompute:

    def test_zero_amplitude_gives_identity(self, propagator, zero_system):
        result = compute_single(propagator, zero_system, u=0.0, dt=0.5)
        assert np.allclose(result, np.eye(2), atol=1e-12)

    def test_unitarity(self, propagator, x_system):
        result = compute_single(propagator, x_system, u=1.0, dt=0.5)
        assert np.allclose(result @ result.conj().T, np.eye(2), atol=1e-12)

    def test_half_rotation(self, propagator, x_system):
        # θ = u·dt/2 = π/4  →  U = (1/√2)(I − i·σₓ) 
        dt = 0.5
        u = np.pi / (2 * dt)
        s = 1 / np.sqrt(2)

        result = compute_single(propagator, x_system, u=u, dt=dt)
        expected = np.array([[ s,    -1j*s],
                              [-1j*s,  s  ]])

        assert np.allclose(result, expected, atol=1e-12), (
            f"Half-rotation failed.\nGot:\n{result}\nExpected:\n{expected}"
        )

    def test_full_X_rotation(self):
        pass

    def test_composition(self, propagator, x_control_system):
        # U(u, 2dt) = U(u, dt)²
        # Test that matrix exponential is self-consistent,
        # i.e., that matrix product of two single-slice propagators equals
        # the two-step propagator
        res_single_dt = test_propagator.compute(
            test_closed_system, test_n_slices, test_control_amplitudes_generic, test_dt
        )[0]

        res_double_dt = test_propagator.compute(
            test_closed_system, test_n_slices, test_control_amplitudes_generic, 2*test_dt
        )[0]

        assert np.allclose(res_double_dt, (res_single_dt @ res_single_dt), atol=1e-14)



