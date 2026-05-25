import numpy as np

from qutip import basis, identity, sigmax, sigmaz, ket2dm

from qoc.objectives.state_transfer import StateTransfer
from qoc.performance.state_fidelity import StateFidelity

from qoc.objectives.gate_synthesis import GateSynthesis
from qoc.performance.gate_fidelity import GateFidelity

from qoc.performance.base import FunctionalPerformanceMeasure

from qoc.systems.closed import ClosedSystem

H0 = 0 * sigmaz()
H_c = sigmax() / 2

system = ClosedSystem(H0=H0, H_controls=[H_c])

times = np.linspace(0, 1, 100)
pi_pulse = np.full((1, 100), np.pi)


# State transfer objective: pure states


objective = StateTransfer(initial=basis(2, 0), target=basis(2, 1), performance_measure=StateFidelity())

final = system.evolve(objective.initial, pi_pulse, times)

print(objective.compute(final))


# State transfer: mixed states

rho_initial = ket2dm(basis(2, 0))
rho_target = ket2dm(basis(2, 1))

objective_mixed = StateTransfer(
        initial=rho_initial,
        target=rho_target,
        performance_measure=StateFidelity())

final_mixed = system.evolve(objective_mixed.initial, pi_pulse, times)
print(objective_mixed.compute(final_mixed))

# Gate synthesis

objective_gate = GateSynthesis(
        initial=identity(2),
        target=sigmax(),
        performance_measure=GateFidelity())

final_gate = system.evolve(objective_gate.initial, pi_pulse, times)
print(objective_gate.compute(final_gate))


# Custom performance measure via FunctionalPerformanceMeasure
# e.g. Hilbert-Schmidt inner product


def hs_fidelity(current, target):
    overlap = (target.dag() * current).tr()
    d = current.shape[0]
    return float(abs(overlap) ** 2 / d ** 2)


# Plug it into GateSynthesis objective

objective_custom_measure = GateSynthesis(
        initial=identity(2),
        target=sigmax(),
        performance_measure=FunctionalPerformanceMeasure(hs_fidelity))

print(objective_custom_measure.compute(final_gate))
