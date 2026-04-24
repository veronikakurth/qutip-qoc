import qutip as qt
from base import Objective
from state_transfer import state_transfer_fidelity
from gate_synthesis import gate_synthesis_fidelity
from qutip.qip.operations import rx

# One possible way to construct Objective and compute how close the system's state is to the target
# Requires the user to provide a function that does the computation. 
# For us it might be difficult to validate it but it is flexible.
if __name__ == '__main__':

    # State transfer

    current = qt.basis(2, 0)
    target = qt.basis(2, 1)
    objective = Objective(target, state_transfer_fidelity) 
    fidelity = objective.compute(current)
    print(fidelity)


    # Gate synthesis
    
    # Ideal Hadamard gate (1 qubit)
    H_ideal = qt.gates.hadamard_transform(1)
    # Perturbed Hadamard: ideal + small random unitary rotation
    theta = 0.01
    U_pert = H_ideal * rx(theta)
    # Compute fidelity between two unitaries
    objective_gate = Objective(H_ideal, gate_synthesis_fidelity)
    gate_fidelity = objective_gate.compute(U_pert)
    print(fidelity)
