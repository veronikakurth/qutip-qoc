from qutip import Qobj
from abc import ABC, abstractmethod

# Arguments for being explicit w.r.t. objective (it may be hard to infer it from initial and target in case of state transfer for open systems (dynamiics described by density matrix) and GateSynthesis for any system (described by unitary)

# Here, it feels natural to talk about fidelity (or another measure of closeness to the target) and make it a part of Objective. In itself, the measure may involve target, so passing a target can be optional. 

# Class for validation
class Objective(ABC):
    """
    Holds the initial condition, the target, and the measure that
    scores how close the current system's state is to the target.

    The concrete type (StateTransfer vs GateSynthesis) determines what
    initial/target must be - validation lives in the subclasses.
    """

    def __init__(self, initial: Qobj, target: Qobj = None):
        self.initial = initial
        self.target = target

    @abstractmethod
    def compute(self, current: Qobj, target: Qobj = None) -> float:
        raise NotImplemented()
