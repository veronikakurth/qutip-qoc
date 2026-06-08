import pytest
import numpy as np
from qutip import sigmax, sigmay, sigmaz

from qoc.dynamics.propagator import StepPropagator
from qoc.systems.closed import ClosedSystem


@pytest.fixture
def test_propagator():
    return StepPropagator()


@pytest.fixture
def test_zero_closed_system():
    return ClosedSystem(H0=0 * sigmax(), H_controls=[0 * sigmay(), 0 * sigmaz()])

@pytest.fixture
def test_end_time():
    return 10


@pytest.fixture
def test_n_slices():
    return 10


@pytest.fixture
def test_dt(test_end_time, test_n_slices):
    return float(test_end_time / test_n_slices)


@pytest.fixture
def test_times(test_end_time, test_n_slices):
    return np.linspace(0, test_end_time, test_n_slices)

@pytest.fixture
def test_control_amplitudes(test_times):
    return np.array([[np.pi for t in test_times] for i in range(2)])


class TestStepPropagatorCompute:

    def test_zero_Hamiltonian(
        self, test_propagator, test_zero_closed_system, test_control_amplitudes, test_n_slices, test_dt
    ):
        # Output must be identity
        result = test_propagator.compute(
            test_zero_closed_system, test_n_slices, test_control_amplitudes, test_dt
        )[-1]
        assert np.allclose(result, np.identity(2))

    def test_unitarity(self):
        pass

    def test_half_angle(self):
        pass

    def test_composition(self):
        # Test that matrix exponential is consistent
        pass
