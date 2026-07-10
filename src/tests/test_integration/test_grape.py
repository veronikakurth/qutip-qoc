import pytest
import numpy as np
import qutip
from qoc.algorithms.grape import GRAPE, adj
from qoc.systems.controlled_system import ControlledSystem
from qoc.problem import OptimalControlProblem
from qoc.objectives import StateTransfer
from qoc.pulse import PiecewiseConstant

@pytest.fixture
def x_closed_system():
    return ControlledSystem(H0=0 * qutip.sigmax(), H_controls=[qutip.sigmax()], is_closed=True)

def test_state_transfer_single_qubit(x_closed_system):
    # Testing GRAPE for a state transfer on a single qubit for which
    # an analytical solution is known
    system = x_closed_system
    T = 10 
    N = 10
    times = np.linspace(0, T, N, endpoint=False)
    np.random.seed(0)

    K = system.dynamics.n_controls
    initial_pulse = np.random.uniform(-0.1, 0.1, (K, N))
    param = PiecewiseConstant(K, times=times)
    dt = param.dt

    initial_state = qutip.basis(2, 0)
    target_state = qutip.basis(2, 1)
    # By default, state fidelity is used as a performance measure in a state transfer task
    objective = StateTransfer(initial_state, target_state)
    control_problem = OptimalControlProblem(system, objective)
    algorithm = GRAPE(parameterization=param)
    result = algorithm.solve(control_problem, initial_amplitudes=initial_pulse)
    assert result.fidelity > 1.0 - 1e-4
    pulse_area = np.sum(result.optimized_pulses) * dt
    reduced = (pulse_area + np.pi) % (2*np.pi) - np.pi # reduce to (-pi, pi]
    assert np.isclose(abs(reduced), np.pi/2, atol=1e-2)


def test_amplitude_bounds_are_enforced(x_closed_system):
    """With a tight amplitude_range, no optimized amplitude exceeds the bound
    and the bound is saturated somewhere (otherwise the test is vacuous)."""
    system = x_closed_system
    T = 10
    N = 10
    np.random.seed(0)

    K = system.dynamics.n_controls
    # Unconstrained optimum amplitude for a pi-pulse is ~pi/T ~= 0.314 — set
    # a tight bound below that so the optimizer is forced into saturation.
    bound = 0.05
    # Non-zero initial guess: the loss landscape has a saddle at u=0 because
    # <target|psi(T)> = 0 makes the adjoint gradient vanish there.
    initial_pulse = np.random.uniform(-0.01, 0.01, (K, N))
    param = PiecewiseConstant(K, T, N, amplitude_range=(-bound, bound))

    objective = StateTransfer(qutip.basis(2, 0), qutip.basis(2, 1))
    problem = OptimalControlProblem(system, objective)
    result = GRAPE(parameterization=param).solve(problem, initial_amplitudes=initial_pulse)

    u = result.optimized_pulses
    assert np.all(u <= bound + 1e-10), f"upper bound violated: max={u.max()}"
    assert np.all(u >= -bound - 1e-10), f"lower bound violated: min={u.min()}"
    assert np.max(np.abs(u)) >= bound - 1e-6, (
        "bound was never saturated — test is vacuous, the bound is too loose"
    )
