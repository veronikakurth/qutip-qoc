from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
from qutip import Qobj, mesolve, sesolve

from qoc.systems.base import System
from qoc.systems.closed import ClosedSystem
from qoc.systems.open import OpenSystem


@dataclass
class EvolutionResult:
    final: Qobj
    trajectory: list[Qobj] | None = None


class Simulator(ABC):

    @abstractmethod
    def evolve(
        self,
        system: System,
        control_amplitudes: np.ndarray,
        times: np.ndarray,
        initial: Qobj,
        *,
        return_trajectory: bool = False,
    ) -> EvolutionResult: ...


class SESolveSim(Simulator):

    def __init__(self, options: dict | None = None):
        self.options = options or {}

    def evolve(self, system, control_amplitudes, times, initial, *, return_trajectory=False):
        H = system.build_hamiltonian(control_amplitudes)
        result = sesolve(H, initial, times, options=self.options)
        return EvolutionResult(
            final=result.states[-1],
            trajectory=result.states if return_trajectory else None,
        )


class MESolveSim(Simulator):

    def __init__(self, options: dict | None = None):
        self.options = options or {}

    def evolve(self, system, control_amplitudes, times, initial, *, return_trajectory=False):
        H = system.build_hamiltonian(control_amplitudes)
        result = mesolve(H, initial, times, c_ops=system.c_ops, options=self.options)
        return EvolutionResult(
            final=result.states[-1],
            trajectory=result.states if return_trajectory else None,
        )


def default_simulator_for(system: System) -> Simulator:
    if isinstance(system, ClosedSystem):
        return SESolveSim()
    if isinstance(system, OpenSystem):
        return MESolveSim()
    raise TypeError(f"No default Simulator for system type {type(system).__name__}")
