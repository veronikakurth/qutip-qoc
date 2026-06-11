import pytest
import numpy as np
import qutip
from qoc.solvers.grape import GRAPE
from qoc.systems.controlled_system import ControlledSystem
from qoc.problem import OptimalControlProblem
from qoc.objectives import StateTransfer

@pytest.fixture
def x_closed_system():
    return ControlledSystem('closed', H0=0 * qutip.sigmax(), H_controls=[qutip.sigmax()])

def check_pulse_area():
    pass

def test_state_transfer_single_qubit(x_closed_system):
    # Testing GRAPE for a state transfer on a single qubit for which
    # an analytical solution is known
    system = x_closed_system
    T = 10 
    N = 100
    dt = T / N
    times = np.linspace(0, T, N)
    initial_pulse = np.random.uniform(0, 0.0001, (system.dynamics.n_controls, N))
    initial_state = qutip.basis(2, 0)
    target_state = qutip.basis(2, 1)

    objective = StateTransfer(initial_state, target_state) # state fidelity will be used as performance measure
    control_problem = OptimalControlProblem(system, objective, times, initial_pulse)
    algorithm = GRAPE()
    result = algorithm.solve(control_problem)
    assert np.allclose(result.fidelity, 1.0 - 1e-4)
    correct_pulse_values = np.array([np.pi / (2 * T)] * N)
    assert np.allclose(result.optimized_pulses, correct_pulse_values)
    

