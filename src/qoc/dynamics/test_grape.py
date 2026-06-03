import numpy as np
from qutip import sigmax, sigmaz, Qobj, basis
from qoc.systems.closed import ClosedSystem
from qoc.objectives.state_transfer import StateTransfer
from qoc.performance.state_fidelity import StateFidelity
import numpy as np
from scipy.linalg import expm

# Discretize time (is already discretized when a times array is passed)
# Compute H which is constant on each slice

def eigendecomposition(H: Qobj, dt: float) -> np.ndarray:
    U_j = expm(-1j * H.full() * dt)
    return U_j

def eigendecomposition_small_n(H: Qobj, dt: float) -> np.ndarray:
    eigvals, V = np.linalg.eigh(H.full())
    U_j = V @  np.diag(np.exp(-1j * eigvals * dt)) @ V.conj().T
    return U_j


def test_initial_propagator_is_identity():
    pass

def test_propagators_are_unitary():
    pass

# Computational cycle of GRAPE

def forward_pass(system: ClosedSystem, initial_state: Qobj, control_amplitudes, N, dt) -> list:
    # Returns: propagators and forward evolution
    propagators = [None] * N

    # forward_evolution = [np.eye(system.shape[0], dtype=complex)] # For gate synthesis
    # We know the size of the array in advance - pre-allocate memory
    forward_evolution = [None] * (N + 1)
    forward_evolution[0] = initial_state.full().copy()

    for j in range(N):
        # Build a full system Hamiltonian for time j (remember: controls are time dependent)
        slice_Hamiltonian = system.build_hamiltonian_time_j(control_amplitudes, j)
        # Compute a propagator for the jth time slice (corresponds to (j, j + 1) time interval)
        slice_propagator = eigendecomposition(slice_Hamiltonian, dt)
        # Collect a propagator into the list
        propagators[j] = slice_propagator # Compute forward evolution up to slice j

        forward_evolution[j + 1] = slice_propagator @ forward_evolution[j]

    return propagators, forward_evolution

def adj(A):
    return np.conj(A).T

def backward_pass(propagators: list, target_state: Qobj):
    # Propagators must be ordered from U_1 to U_N
    N = len(propagators)
    co_states = [None] * (N + 1)
    co_states[N] = target_state.full()

    for j in range(N):
        sub_co_state = co_states[-1]
        for i in reversed(range(N, j)):
            sub_co_state = adj(propagators[i]) @ sub_co_state
        co_states[j] = sub_co_state
    return co_states

def grape_gradient(forward_evolution: list, co_states: list, N: int, controls, system: ClosedSystem, dt: float):
    # adjoint method, requires previously computed forward and backward pass
    grads = np.array([[None for t in range(N)] * system.n_controls])
    for j in range(N):
        for k, Hc in enumerate(system.H_controls):
            control_operator = -1j * Hc * dt
            grads[k, j] = 2 * np.real(adj(co_states[j]) @ control_operator.full() @ forward_evolution[j - 1])
    return grads



def control_update(control_amplitudes):
    pass

def grape():
    # Basic case: state transfer objective + pure states

    H0 = 0 * sigmaz()
    H_c = [sigmax() / 2]
    system = ClosedSystem(H0=H0, H_controls=H_c)
    
    T = 10
    N = 10
    dt = T/N
    times = np.linspace(0, T, N)
    control_amplitudes = np.array([[np.pi for t in times] for i in range(system.n_controls)])
    
    initial_state = basis(2, 0)
    target_state = basis(2, 1)
    objective = StateTransfer(initial_state, target_state, performance_measure=StateFidelity())

    slice_propagators, forward_evolution = forward_pass(system, initial_state, control_amplitudes, N, dt)
    print(f"control_amplitudes.shape: {control_amplitudes.shape}") 
    print(f"Initial state: \n {initial_state}")

    co_states = backward_pass(slice_propagators, target_state)
    for j in range(len(slice_propagators)):
        print(f"Slice propagator at j={j}: \n {slice_propagators[j]}")
        print(f"Forward step at j = {j}: \n {forward_evolution[j]} \n")
        print(f"Co-state at j = {j}: \n {co_states[j]} \n")
    print(len(slice_propagators))
    print(len(forward_evolution))
    print(len(co_states))

    grads = grape_gradient(forward_evolution, co_states, N, control_amplitudes, system, dt)
    print(f"grads: \n {grads}")
    print(grads.shape)

if __name__ == '__main__':
    grape()
