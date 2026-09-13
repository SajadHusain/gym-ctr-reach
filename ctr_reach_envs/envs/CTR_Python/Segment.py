import numpy as np


class Segment:
    """Segment three overlapping tubes into intervals of constant properties."""

    def __init__(self, t1, t2, t3, base, *, quantize=True):
        stiffness = np.array([t1.E, t2.E, t3.E])
        torsion = np.array([t1.G, t2.G, t3.G])
        curve_x = np.array([t1.U_x, t2.U_x, t3.U_x])
        curve_y = np.array([t1.U_y, t2.U_y, t3.U_y])

        d_tip = np.array([t1.L, t2.L, t3.L]) + base
        d_c = d_tip - np.array([t1.L_c, t2.L_c, t3.L_c])
        points = np.hstack((0.0, base, d_c, d_tip))
        index = np.argsort(points)
        raw_length = np.diff(np.sort(points))
        segment_length = 1e-5 * np.floor(1e5 * raw_length) if quantize else raw_length
        # Derivatives within a fixed event ordering. The legacy floor has zero
        # derivative inside each cell; it cannot use the continuous derivative.
        point_gradients = np.vstack((np.zeros((1, 3)), np.tile(np.eye(3), (3, 1))))
        raw_gradients = np.diff(point_gradients[index], axis=0)
        moving = np.any(raw_gradients != 0, axis=1)
        length_gradients = np.zeros_like(raw_gradients) if quantize else raw_gradients
        self.derivative_valid = not np.any(moving & (raw_length < 1e-9))
        self.derivative_reason = "" if self.derivative_valid else "moving segment boundary"
        if quantize and np.any(moving & (abs(raw_length / 1e-5 - np.round(raw_length / 1e-5)) * 1e-5 < 1e-9)):
            self.derivative_valid = False
            self.derivative_reason = "legacy rounding boundary"

        e = np.zeros((3, segment_length.size))
        g = np.zeros((3, segment_length.size))
        u_x = np.zeros((3, segment_length.size))
        u_y = np.zeros((3, segment_length.size))

        for i in range(3):
            a = np.where(index == i + 1)[0].item()
            b = np.where(index == i + 4)[0].item()
            c = np.where(index == i + 7)[0].item()
            if segment_length[a] == 0:
                a += 1
            if segment_length[b] == 0:
                b += 1
            if segment_length[a] == 0:
                a += 1
            if c <= segment_length.size - 1 and segment_length[c] == 0:
                c += 1
            e[i, np.arange(a, c)] = stiffness[i]
            g[i, np.arange(a, c)] = torsion[i]
            u_x[i, np.arange(b, c)] = curve_x[i]
            u_y[i, np.arange(b, c)] = curve_y[i]

        nonzero = segment_length != 0.0
        length = segment_length[nonzero]
        ee = e[:, nonzero]
        gg = g[:, nonzero]
        uu_x = u_x[:, nonzero]
        uu_y = u_y[:, nonzero]
        length_sum = np.cumsum(length)
        keep = length_sum + min(base) > 0
        self.S = length_sum[keep] + min(base)
        endpoint_gradients = np.cumsum(length_gradients[nonzero], axis=0) + np.eye(3)[np.argmin(base)]
        endpoints = length_sum + min(base)
        if np.any((abs(endpoints) < 1e-9) & np.any(endpoint_gradients != 0, axis=1)):
            self.derivative_valid = False
            self.derivative_reason = "template crossing"
        self.S_beta = endpoint_gradients[keep]

        e_t = ee[:, keep]
        g_t = gg[:, keep]
        self.EI = (e_t.T * np.array([t1.I, t2.I, t3.I])).T
        self.GJ = (g_t.T * np.array([t1.J, t2.J, t3.J])).T
        self.U_x = uu_x[:, keep]
        self.U_y = uu_y[:, keep]
