import numpy as np


@np.vectorize
def harmonic_potential(x):
    return 0.5 * x * x


def search_inversion_point(vpot):
    ediff = vpot[::] - 1.  # corr to V(x) - E
    ediff[np.isclose(ediff, 0)] = 1e-20

    icl = -1
    for i in range(len(vpot) - 1):
        if np.sign(ediff[i]) != np.sign(ediff[i + 1]):
            icl = i

    assert(icl != -1)
    return icl


def numerov(xmax, mesh, nodes, n_iter):
    """
    solve one-dimention Schrodinger equation with Numerov's method

    assume potential is even w.r.t. x = 0 !!!
    """
    dx = xmax / mesh
    ddx12 = dx * dx / 12.

    hnodes = nodes // 2  # hnodes is the number of nodes in x > 0 region

    x = np.linspace(0, xmax, num=mesh + 1)  # meshed coordinates
    vpot = harmonic_potential(x)  # potential energy

    eigval_ub = max(vpot)  # upper bound of eigenvalue
    eigval_lb = min(vpot)  # lower bound of eigenvalue
    eigval = 0.5 * (eigval_ub + eigval_lb)

    # search eigenvalue(energy) by binary search
    for _ in range(n_iter):
        if eigval_ub - eigval_lb < 1e-10:
            break
        f = 1. + ddx12 * (eigval - vpot)  # for Numerov method

        # search classical turning point
        icl = search_inversion_point(vpot)
        if icl >= mesh - 2:
            raise ValueError("last change of sign too far")
        elif icl < 1:
            raise ValueError("no classical turning point?")

        # initial values
        y = np.zeros(mesh + 1)
        if nodes % 2 == 0:
            y[0] = 1.  # arbitrary
            y[1] = 0.5 * (12. - f[0] * 10.) * y[0] / f[1]  # eq(1.33)
        else:
            y[0] = 0.
            y[1] = dx  # arbitrary

        # forward integration
        ncross = 0
        for i in range(1, mesh):
            y[i + 1] = ((12. - 10. * f[i]) * y[i] - f[i - 1] * y[i - 1]) / f[i + 1]
            if np.sign(y[i]) != np.sign(y[i + 1]):
                ncross += 1

        if ncross > hnodes:
            # too many crossing points mean too high energy
            eigval_ub = eigval
        else:
            eigval_lb = eigval
        # update trial eigenvalue
        eigval = 0.5 * (eigval_ub + eigval_lb)

    return x, y, eigval


if __name__ == '__main__':
    xmax = 10.
    mesh = 100
    nodes = 0
    n_iter = 1000

    x, y, eigval = numerov(xmax, mesh, nodes, n_iter)
    import matplotlib.pyplot as plt
    plt.plot(x, y)
    plt.show()
    print(y)
    print(eigval)
