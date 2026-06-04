from abc import ABC, abstractmethod

class OptimizerResult:
    pass

class Optimizer(ABC):

    @abstractmethod
    def minimize(self, **kwargs) -> OptimizerResult:
        pass
