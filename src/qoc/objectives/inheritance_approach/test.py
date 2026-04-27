import qutip as qt
from fidelity_objective import FidelityObjective
from gate_synthesis import GateFidelityObjective
from qutip.qip.operations import rx


def mixture(states, probs):
    return sum(p * qt.ket2dm(s) for s, p in zip(states, probs))

# One possible way to use a specific Objective and compute how close the system's state is to the target
# Requires the user to provide a function that does the computation. 
# For us it might be difficult to validate it but it is flexible.
if __name__ == '__main__':

    # State transfer (pure state)
    current = qt.basis(2, 0)
    target = qt.basis(2, 1)
    objective_pure = FidelityObjective(target) 
    fidelity_pure = objective_pure.compute(current)
    print(f"Fidelity for pure state transfer:\n", fidelity_pure)
    # State transfer (mixed state)
    mixed_state_target = mixture([qt.basis(2, 0), qt.basis(2, 1)], [0.6, 0.4])
    mixed_state_current = mixture([qt.basis(2, 0), qt.basis(2, 1)], [0.5, 0.5])

    objective_mixed = FidelityObjective(mixed_state_target)
    fidelity_mixed = objective_mixed.compute(mixed_state_current)
    print(f"Fidelity for mixed state transfer:\n",fidelity_mixed)

    # Gate synthesis
    # Ideal Hadamard gate (1 qubit)
    H_ideal = qt.gates.hadamard_transform(1)
    # Perturbed Hadamard: ideal + small random unitary rotation
    theta = 0.01
    U_pert = H_ideal * rx(theta)
    # Compute fidelity between two unitaries
    objective_gate = GateFidelityObjective(H_ideal)
    gate_fidelity = objective_gate.compute(U_pert)
    print("Gate fidelity: \n", gate_fidelity)
