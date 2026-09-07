from math import pi


class Tube:
    """Physical properties of one concentric tube, in SI units."""

    def __init__(
        self,
        length,
        length_curved,
        diameter_inner,
        diameter_outer,
        stiffness,
        torsional_stiffness,
        x_curvature,
        y_curvature,
    ):
        values = {
            "length": length,
            "length_curved": length_curved,
            "diameter_inner": diameter_inner,
            "diameter_outer": diameter_outer,
            "stiffness": stiffness,
            "torsional_stiffness": torsional_stiffness,
        }
        if any(float(value) <= 0.0 for value in values.values()):
            raise ValueError(f"Tube dimensions and stiffnesses must be positive: {values}")
        if length_curved > length:
            raise ValueError("length_curved cannot exceed length")
        if diameter_inner >= diameter_outer:
            raise ValueError("diameter_inner must be smaller than diameter_outer")

        self.L = float(length)
        self.L_c = float(length_curved)
        self.L_s = self.L - self.L_c
        self.diameter_inner = float(diameter_inner)
        self.diameter_outer = float(diameter_outer)
        diameter_term = self.diameter_outer**4 - self.diameter_inner**4
        self.J = pi * diameter_term / 32.0
        self.I = pi * diameter_term / 64.0
        self.E = float(stiffness)
        self.G = float(torsional_stiffness)
        self.U_x = float(x_curvature)
        self.U_y = float(y_curvature)

