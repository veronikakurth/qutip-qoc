class Objective:
    """
    Holds the target and the measure scoring how close the current system's state is to the target.

    Design approach: composition (function injection) - a user may plug and play with custom objectives.

    """

    def __init__(self, initial, target, objective_func):
        self.initial = initial
        self.target = target
        self.objective_func = objective_func

    def compute(self, current, target):
        return self.objective_func(current, target)
