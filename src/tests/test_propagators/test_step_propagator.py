import pytest
import numpy as np
from qutip import sigmax, sigmay, sigmaz

from qoc.dynamics.propagator import StepPropagator
from qoc.systems.closed import ClosedSystem


@pytest.fixture
def test_propagator():
    return StepPropagator()


@pytest.fixture
def test_closed_system():
    return ClosedSystem(H0=0 * sigmax(), H_controls=[0 * sigmay(), 0 * sigmaz()])
@pytest.fixture
def test_end_time():
    return 10


@pytest.fixture
def test_n_slices():
    return 1


@pytest.fixture
def test_dt(test_end_time, test_n_slices):
    return float(test_end_time / test_n_slices)


@pytest.fixture
def test_times(test_end_time, test_n_slices):
    return np.linspace(0, test_end_time, test_n_slices)

@pytest.fixture
def test_control_amplitudes(test_times):
    return np.array([[np.pi for t in test_times] for i in range(2)])

@pytest.fixture
def test_control_amplitudes_generic(test_times):
    return np.array([[1.0 for t in test_times] for i in range(2)])

# TODO: once StepPropagator contains propagation function for a single slice, refactor tests - don't use indexing of the lsit
class TestStepPropagatorCompute:

    def test_zero_Hamiltonian(
        self, test_propagator, test_closed_system, test_control_amplitudes, test_n_slices, test_dt
    ):
        # Output must be identity
        result = test_propagator.compute(
            test_closed_system, test_n_slices, test_control_amplitudes, test_dt
        )[0]
        rows_n, _ = test_closed_system.shape
        assert np.allclose(result, np.identity(n=rows_n))

    def test_unitarity(
        self, test_propagator, test_closed_system, test_control_amplitudes, test_n_slices, test_dt
    ):
        result = test_propagator.compute(
            test_closed_system, test_n_slices, test_control_amplitudes, test_dt
        )[0]

        rows_n, _ = test_closed_system.shape
        assert np.allclose(result @ result.conj().T, np.identity(rows_n))

    def test_half_rotation(self):
        pass

    def test_full_X_rotation(self):
        pass

    def test_composition(self, test_propagator, test_closed_system, test_n_slices, test_control_amplitudes_generic, test_dt):
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

        assert np.allclose(res_double_dt, (res_single_dt @ res_double_dt), atol=1e-14)



