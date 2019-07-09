import numpy as np


class Hartree:
    """
    solve Hartree equation for ground state of Helium-like atom.

    Parameters
    ----------
    atomic_charge: int
    beta: float, mixing parameter for self-consistent iterations
    tol: float, tolerance for SCF
    """
    def __init__(self, atomic_charge, beta, tol):
        self.atomic_charge = atomic_charge
        assert(self.atomic_charge >= 2)
        self.beta = beta
        assert(0 <= self.beta <= 1.0)
        self.tol = tol
        assert(tol >= 0)

        self.n_ele = 2  # He has 2 electorns

        self.sss = SphericalSchrodingerSolver(self.atomic_charge,
                                              rmax=100., xmin=-6., dx=1e-2)

    @property
    def dx(self):
        return self.sss.dx

    def solve(self):
        # ground state: (1s)^2
        n = 1
        l = 0

        # initial potential
        vpot = coulomb_potential(self.sss.r, self.atomic_charge)  # eq (6.21)
        vscf = vpot[::]

        itr = 1
        while True:
            print("#" * 80)
            print(f" SCF iteration # {itr}")

            energy, y = self.sss.solve(vscf, n, l)
            rho = self._charge_density(y)
            ehx, vhx = self._hartree_potential(rho, self.sss.dx)

            # variation
            v_delta = vpot + vhx - vscf
            de = np.sum(v_delta * rho * (self.sss.r ** 3)) * 4 * np.pi * self.dx

            e_eigval = 2 * energy
            e_vion = np.sum(vpot * rho * self.sss.r ** 3) * 4 * np.pi * self.dx
            e_kin = e_eigval - np.sum(vscf * rho * self.sss.r ** 3) * 4 * np.pi * self.dx

            e_tot = e_eigval - ehx + de
            e_tot1 = e_kin + e_vion + ehx

            print(f"eigenvalue={energy}")
            print(f"Eigenvalue energy:    {e_eigval}")
            print(f"Kinetic energy:       {e_kin}")
            print(f"External pot. energy: {e_vion}")
            print(f"Hartree+Exch. energy: {ehx}")
            print(f"Variational correct.: {de}")
            print(f"Total energy:         {e_tot} {e_tot1}")

            virial = -(e_vion + ehx) / e_kin
            print(f"Virial check: {virial}")

            if de < self.tol:
                break

            # update potential
            vscf = vscf + self.beta * v_delta

            itr += 1

    def _charge_density(self, eigfunc):
        """
        compute the charge density rho(r) of a He-like atom from eigenfunction in logarithmic grid
        """
        rho = self.n_ele * (eigfunc ** 2) / (4. * np.pi * self.sss.r)
        return rho

    def _hartree_potential(self, rho, dx):
        qe2 = 2.0

        # eq (6.21), (6.22)
        charge = np.cumsum(4 * np.pi * dx * rho * (self.sss.r ** 3))
        vhx = qe2 * charge / (self.sss.r ** 2)

        tot_charge = charge[-1]
        print(f"Total charge = {tot_charge}")

        vhx_shift = np.zeros_like(vhx)
        vhx_shift[-1] = qe2 * tot_charge / self.sss.r[-1]
        for i in range(self.sss.mesh - 1, 0, -1):
            vhx_shift[i] = vhx_shift[i + 1] + vhx[i] * self.sss.r[i] * dx

        ehx = np.sum(vhx_shift * rho * (self.sss.r ** 3)) * 4 * np.pi * dx
        ehx /= 2

        # TODO
        ehx *= 0.5
        vhx_shift = 0.5 * vhx_shift

        return ehx, vhx_shift


class SphericalSchrodingerSolver:
    """
    solve Schr\"odinger equation under centrifugal potential

    We use "Rydberg" atomic units:
        units of length are Bohr radii a_0,
        and units of energies are Ry(=13.6058...eV)
    In this atomic units, hbar = 1, m_e = 1 / 2, q_e^2 = 2

    Parameters
    ----------
    atomic_charge: int
        Z
    rmax: double
        upper bound of calculated radii
    xmin: double
        lower bound of calculated logarithmic grid
    dx: double
        step size of logarithmic grid
    """
    def __init__(self, atomic_charge, rmax=100., xmin=-6., dx=1e-2):
        self.atomic_charge = atomic_charge  # Z
        self.rmax = rmax
        self.xmin = xmin
        self.dx = dx

        self.eps = 1e-10
        self.n_iter = 100

        self.ddx12 = self.dx * self.dx / 12

        # number of grid points
        self.mesh = int((np.log(self.atomic_charge * rmax) - xmin) / dx)
        assert(self.mesh > 0)

        # generate mesh for logarithmic grid
        self.x = np.linspace(self.xmin, self.xmin + self.dx * self.mesh, num=self.mesh + 1)
        # generate mesh for real radii
        self.r = np.exp(self.x) / self.atomic_charge

    def solve(self, vpot, n, l):
        """
        find eigenvalue and wavefunction for principal quantum number n and angular momentum l.
        """
        assert(n >= 1 and l >= 0 and n > l)

        # search eigenvalue(energy) by binary search
        eigval_ub = vpot[-1]  # upper bound of eigenvalue
        eigval_lb = min(vpot + ((l + 0.5) / self.r) ** 2)  # lower bound of eigenvalue
        assert(eigval_ub >= eigval_lb)
        eigval = 0.5 * (eigval_ub + eigval_lb)
        eigfunc = None

        for _ in range(self.n_iter):
            if eigval_ub - eigval_lb < 1e-10:
                break
            # for Numerov method, eq (1.31)
            f = 1. + self.ddx12 * ((eigval - vpot) * self.r * self.r - (l + 0.5) ** 2)

            # search classical turning point
            # import pdb; pdb.set_trace()
            icl = search_turning_point(f - 1.)
            if icl >= self.mesh - 2:
                raise ValueError("last change of sign too far")
            elif icl < 1:
                raise ValueError("no classical turning point?")

            # outward integration
            y = self._outward_numerov(f, l, icl)

            # number of crossings
            nodes = n - l - 1
            ncross = number_of_crossings(y[:icl + 1])

            # WIP below
            if ncross > nodes:
                # too many crossing points mean too high energy
                eigval_ub = eigval
                eigval = 0.5 * (eigval_ub + eigval_lb)
            elif ncross < nodes:
                # too short crossing points mean too low energy
                eigval_lb = eigval
                eigval = 0.5 * (eigval_ub + eigval_lb)
            else:
                # if number of crossing is correct, proceed to inward integration

                # inward integration
                y_icl_prev = y[icl]
                y = self._inward_numerov(y, f, icl)

                # rescale function to match at the classical turning point(icl)
                scaling = y_icl_prev / y[icl]
                y[icl:] *= scaling

                # normalize wavefunction y on the segment
                norm = np.sqrt(np.sum(y[1:] * y[1:] * self.r[1:] * self.r[1:]) * self.dx)
                y /= norm

                # find the value of the casp, eq (1.32)
                ycusp = (y[icl - 1] * f[icl - 1] + y[icl + 1] * f[icl + 1] + 10. * y[icl] * f[icl]) / 12.
                dfcusp = f[icl] * (y[icl] / ycusp - 1.)
                delta_e = dfcusp / self.ddx12 * ycusp * ycusp * self.dx
                if delta_e > 0:
                    eigval_lb = eigval
                elif delta_e < 0:
                    eigval_ub = eigval

                eigval += delta_e
                eigval = min(eigval_ub, max(eigval_lb, eigval))

            eigfunc = y

        return eigval, eigfunc

    def _outward_numerov(self, f, l, icl):
        """
        outward integration in [0, icl]
        """
        y = np.zeros(self.mesh + 1)
        # initial value
        y[0] = (self.r[0] ** (l + 1)) \
            * (1. - 2. * self.atomic_charge * self.r[0] / (2. * l + 2.)) / np.sqrt(self.r[0])
        y[1] = (self.r[1] ** (l + 1)) \
            * (1. - 2. * self.atomic_charge * self.r[1] / (2. * l + 2.)) / np.sqrt(self.r[1])

        # outward integration
        for i in range(1, icl):
            y[i + 1] = ((12. - 10. * f[i]) * y[i] - f[i - 1] * y[i - 1]) / f[i + 1]  # eq (1.32)
        return y

    def _inward_numerov(self, y, f, icl):
        """
        inward integration in [icl, mesh]
        """
        # initial values
        y[-1] = self.dx  # arbitrary
        y[-2] = (12. - 10. * f[-1]) * y[-1] / f[-2]  # let y[mesh + 1] = 0
        # inward integration
        for i in range(self.mesh - 1, icl, -1):
            y[i - 1] = ((12. - 10. * f[i]) * y[i] - f[i + 1] * y[i + 1]) / f[i - 1]
            # if too big, rescale initial value
            if y[i - 1] > 1e10:
                y[i - 1:] /= y[i - 1]
        return y


def coulomb_potential(r, atomic_charge):
    """
    Coulomb potential in Rydberg atomic units
    """
    return -2. * atomic_charge / r


def search_turning_point(arr):
    arr_tmp = arr[::]
    # prevent from oscillating around 0 to detect true turning point
    arr_tmp[np.isclose(arr_tmp, 0)] = 1e-20

    icl = -1
    for i in range(len(arr) - 1):
        if arr_tmp[i] != np.sign(arr_tmp[i - 1]) * np.abs(arr_tmp[i]):
            icl = i

    assert(icl != -1)
    return icl


def number_of_crossings(y):
    ncross = 0
    for i in range(1, len(y) - 1):
        if y[i] != np.sign(y[i + 1]) * np.abs(y[i]):
            ncross += 1
    return ncross


if __name__ == '__main__':
    atomic_charge = 2
    hartree = Hartree(atomic_charge, 0.1, 1e-4)
    print(hartree.solve())

    """
    import matplotlib.pyplot as plt
    plt.plot(x, y)
    plt.title(f"Harmonic oscillator phi_{nodes}")
    plt.savefig("ho.png")
    """
