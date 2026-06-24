import pytest
import numpy as np
import qutip
import matplotlib.pyplot as plt
from qoc.algorithms.grape import GRAPE, adj
from qoc.systems.controlled_system import ControlledSystem
from qoc.problem import OptimalControlProblem
from qoc.objectives import StateTransfer
from qoc.pulse import PiecewiseConstant

@pytest.fixture
def x_closed_system():
    return ControlledSystem(H0=0 * qutip.sigmax(), H_controls=[qutip.sigmax()], kind='closed')

def test_state_transfer_single_qubit(x_closed_system):
    # Testing GRAPE for a state transfer on a single qubit for which
    # an analytical solution is known
    system = x_closed_system
    T = 10 
    N = 10
    dt = T / N
    times = np.linspace(0, T, N, endpoint=False)
    np.random.seed(0)

    K = system.dynamics.n_controls
    initial_pulse = np.random.uniform(-0.1, 0.1, (K, N))
    param = PiecewiseConstant(K, N)
    theta0 = param.initial_theta(initial_pulse)

    initial_state = qutip.basis(2, 0)
    target_state = qutip.basis(2, 1)
    # By default, state fidelity is used as a performance measure in a state transfer task
    objective = StateTransfer(initial_state, target_state)
    control_problem = OptimalControlProblem(system, objective, times, theta0)
    algorithm = GRAPE(parameterization=param)
    result = algorithm.solve(control_problem)
    assert result.fidelity > 1.0 - 1e-4
    pulse_area = np.sum(result.optimized_pulses) * dt
    reduced = (pulse_area + np.pi) % (2*np.pi) - np.pi # reduce to (-pi, pi]
    assert np.isclose(abs(reduced), np.pi/2, atol=1e-2)
    plt.plot(result.history)
    plt.xlabel("iterations")
    plt.ylabel("fidelity")
    plt.show()
