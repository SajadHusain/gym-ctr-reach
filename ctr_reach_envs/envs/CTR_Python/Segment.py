import numpy as np


class Segment:
    """Segment three overlapping tubes into intervals of constant properties."""

    def __init__(self, t1, t2, t3, base):
        stiffness = np.array([t1.E, t2.E, t3.E])
        torsion = np.array([t1.G, t2.G, t3.G])
        curve_x = np.array([t1.U_x, t2.U_x, t3.U_x])
        curve_y = np.array([t1.U_y, t2.U_y, t3.U_y])

        d_tip = np.array([t1.L, t2.L, t3.L]) + base
        d_c = d_tip - np.array([t1.L_c, t2.L_c, t3.L_c])
        points = np.hstack((0.0, base, d_c, d_tip))
        index = np.argsort(points)
        segment_length = 1e-5 * np.floor(1e5 * np.diff(np.sort(points)))

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

        e_t = ee[:, keep]
        g_t = gg[:, keep]
        self.EI = (e_t.T * np.array([t1.I, t2.I, t3.I])).T
        self.GJ = (g_t.T * np.array([t1.J, t2.J, t3.J])).T
        self.U_x = uu_x[:, keep]
        self.U_y = uu_y[:, keep]
