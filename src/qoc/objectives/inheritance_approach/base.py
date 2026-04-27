from abc import ABC, abstractmethod

class Objective(ABC):
    """
    Holds the target and the measure scoring how close the current system's state is to the target.

    Design approach: composition (function injection) - a user may plug and play with custom objectives.

    """

    def __init__(self, target):
        self.target = target
    
    @abstractmethod
    def compute(self, current):
        pass
