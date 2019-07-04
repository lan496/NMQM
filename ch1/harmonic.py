import numpy as np


@np.vectorize
def harmonic_potential(x):
    return 0.5 * x * x


def search_inversion_point(f):
    ediff = f[::] - 1.  # corr to V(x) - E
    ediff[np.isclose(ediff, 0)] = 1e-20

    icl = -1
    for i in range(len(f) - 1):
        if ediff[i] != np.sign(ediff[i + 1]) * np.abs(ediff[i]):
            icl = i

    assert(icl != -1)
    return icl


def number_of_crossings(y):
    ncross = 0
    for i in range(1, len(y) - 1):
        if y[i] != np.sign(y[i + 1]) * np.abs(y[i]):
            ncross += 1
    return ncross


def outward_numerov(x, f, mesh, nodes, dx, icl):
    """
    outward integration in [0, icl]
    """
    y = np.zeros(mesh + 1)
    # initial value
    if nodes % 2 == 0:
        y[0] = 1.  # arbitrary
        y[1] = 0.5 * (12. - f[0] * 10.) * y[0] / f[1]  # eq(1.33)
    else:
        y[0] = 0.
        y[1] = dx  # arbitrary
    # outward integration
    for i in range(1, icl):
        y[i + 1] = ((12. - 10. * f[i]) * y[i] - f[i - 1] * y[i - 1]) / f[i + 1]  # eq (1.32)
    return y


def inward_numerov(y, f, mesh, dx, icl):
    """
    inward integration in [icl, mesh]
    """
    # initial values
    y[mesh] = dx  # arbitrary
    y[mesh - 1] = (12. - 10. * f[mesh]) * y[mesh] / f[mesh - 1]  # let y[mesh + 1] = 0
    # inward integration
    for i in range(mesh - 1, icl, -1):
        y[i - 1] = ((12. - 10. * f[i]) * y[i] - f[i + 1] * y[i + 1]) / f[i - 1]
    return y


def solve1D(xmax, mesh, nodes, n_iter):
    """
    solve one-dimention Schrodinger equation with Numerov's method

    assume potential is even w.r.t. x = 0 !!!
    """
    dx = xmax / mesh

    hnodes = nodes // 2  # hnodes is the number of nodes in x > 0 region

    x = np.linspace(0, xmax, num=mesh + 1)  # meshed coordinates
    vpot = harmonic_potential(x)  # potential energy

    # search eigenvalue(energy) by binary search
    eigval_ub = max(vpot)  # upper bound of eigenvalue
    eigval_lb = min(vpot)  # lower bound of eigenvalue
    eigval = 0.5 * (eigval_ub + eigval_lb)

    for _ in range(n_iter):
        if eigval_ub - eigval_lb < 1e-10:
            break
        f = 1. + dx * dx / 12. * 2. * (eigval - vpot)  # for Numerov method, eq (1.31)

        # search classical turning point
        icl = search_inversion_point(f)
        if icl >= mesh - 2:
            raise ValueError("last change of sign too far")
        elif icl < 1:
            raise ValueError("no classical turning point?")

        # outward integration
        y = outward_numerov(x, f, mesh, nodes, dx, icl)

        # number of crossings
        ncross = number_of_crossings(y[:icl + 1])

        if ncross > hnodes:
            # too many crossing points mean too high energy
            eigval_ub = eigval
        elif ncross < hnodes:
            # too short crossing points mean too low energy
            eigval_lb = eigval
        else:
            # if number of crossing is correct, proceed to inward integration

            # inward integration
            y_icl_prev = y[icl]
            y = inward_numerov(y, f, mesh, dx, icl)

            # rescale function to match at the classical turning point(icl)
            scaling = y_icl_prev / y[icl]
            y[icl:] *= scaling

            # normalize wavefunction y on the [-xmax, xmax] segment
            norm = np.sum(y[1:] * y[1:])
            norm = np.sqrt(dx * (y[0] * y[0] + 2. * norm))
            y /= norm

            # check discontinuity in the first derivative, eq (1.37)
            jump = (y[icl + 1] + y[icl - 1] - (14. - 12. * f[icl]) * y[icl]) / dx
            if jump * y[icl] > 0:
                eigval_ub = eigval
            else:
                eigval_lb = eigval

        # update trial eigenvalue
        eigval = 0.5 * (eigval_ub + eigval_lb)

    return x, y, eigval


if __name__ == '__main__':
    xmax = 10.
    mesh = 100
    nodes = 2
    n_iter = 1000

    x, y, eigval = solve1D(xmax, mesh, nodes, n_iter)
    import matplotlib.pyplot as plt
    plt.plot(x, y)
    plt.title(f"Harmonic oscillator phi_{nodes}")
    plt.savefig("ho.png")
