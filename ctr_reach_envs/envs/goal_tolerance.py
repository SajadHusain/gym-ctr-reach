class GoalTolerance:
    """Position-tolerance curriculum, with distances measured in metres."""

    def __init__(self, parameters):
        self.initial = float(parameters["initial_tol"])
        self.final = float(parameters["final_tol"])
        self.duration = int(parameters["N_ts"])
        self.function = parameters["function"]
        self.fixed = float(parameters["set_tol"])
        if self.function not in {"constant", "linear", "decay"}:
            raise ValueError("function must be constant, linear, or decay")
        if not 0.0 < self.final <= self.initial:
            raise ValueError("Require 0 < final_tol <= initial_tol")
        if self.duration <= 0:
            raise ValueError("N_ts must be positive")
        self.current = self.fixed if self.fixed > 0.0 else self.initial

    def update(self, timestep: int) -> None:
        if self.fixed > 0.0:
            self.current = self.fixed
            return
        progress = min(max(float(timestep) / self.duration, 0.0), 1.0)
        if self.function == "constant":
            self.current = self.initial
        elif self.function == "linear":
            self.current = self.initial + progress * (self.final - self.initial)
        else:
            self.current = self.initial * (self.final / self.initial) ** progress

    def get_tol(self) -> float:
        return float(self.current)

