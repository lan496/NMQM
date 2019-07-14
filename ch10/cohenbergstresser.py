from itertools import product

import numpy as np
import matplotlib.pyplot as plt


class Zincblende:
    """
    Parameters for pesudopotential of zincblende structure

    Parameters
    ----------
    a: float
        lattice length of primitive fcc (a.u.)
    vsn: float
        (n=3, 8, 11) symmetric term
    vsa: float
        (n=3, 4, 11) antisymmetric term
    """
    def __init__(self, a, vs3, vs8, vs11, va3, va4, va11):
        # lattice constant of primitive fcc in a.u.
        self.a = a

        self.vs3 = vs3
        self.vs8 = vs8
        self.vs11 = vs11
        self.va3 = va3
        self.va4 = va4
        self.va11 = va11

        # two atoms are at -tau and +tau
        self.tau = np.array([0.125, 0.125, 0.125])

        # basis vectors of reciprocal lattice in 2pi/a units
        self.rcp = np.array([[-1, 1, 1], [1, -1, 1], [1, 1, -1]], dtype=np.float)

        self.cell = list(self.a * np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]]))
        self.positions = [list(self.tau), list(-self.tau)]
        self.numbers = [0, 1]

    @property
    def structure(self):
        return (self.cell, self.positions, self.numbers)

    def get_nmax(self, encut):
        nmax = int(np.ceil(np.sqrt(encut) / (2 * np.pi / self.a * np.sqrt(3)))) + 1
        return nmax

    def get_potential_term(self, wavevector):
        g2 = np.sum(wavevector * wavevector)

        eps = 1e-8
        if np.abs(g2 - 3) < eps:
            vsg = self.vs3
            vag = self.va3
        elif np.abs(g2 - 4) < eps:
            vsg = 0
            vag = self.va4
        elif np.abs(g2 - 8) < eps:
            vsg = self.vs8
            vag = 0
        elif np.abs(g2 - 11) < eps:
            vsg = self.vs11
            vag = self.va11
        else:
            vsg = 0
            vag = 0

        return vsg, vag


class Diamond(Zincblende):

    def __init__(self, a, vs3, vs8, vs11):
        super().__init__(a, vs3, vs8, vs11, 0, 0, 0)
        self.numbers = [0, 0]


class Pseudopotential:
    """
    Band structure of zincblende and diamond structure
    Units: Rydberg atomic units

    Refs
    ----
    Cohen and Bergstresser, PRB 141, 789 (1966)
    """
    def __init__(self, zinc: Zincblende, encut):
        self.zinc = zinc
        self.encut = encut
        assert(self.encut > 0)

    @property
    def rcp(self):
        # basis vectors of reciprocal lattice in 2pi/a units
        return self.zinc.rcp

    def solve(self, kpoint):
        # generate plane wave basis
        nmax = self.zinc.get_nmax(self.encut)
        kg = []
        for nn in product(range(-nmax, nmax + 1), repeat=3):
            kn = kpoint + np.dot(self.rcp, nn)
            if np.linalg.norm(self.zinc.a / (2 * np.pi) * kn) < self.encut:
                kg.append(kn)

        npw = len(kg)
        if npw < 1:
            raise ValueError("Incorrect number of plane waves!")
        else:
            print(f"Number of plane waves: {npw}")

        # hamiltonian matrix
        H = np.zeros((npw, npw), dtype=np.complex)
        for i, j in product(range(npw), repeat=2):
            gij = kg[i] - kg[j]
            vsg, vag = self.zinc.get_potential_term(gij)
            if i == j:
                H[i, j] = np.sum((2 * np.pi / self.zinc.a * kg[i]) ** 2) + vsg
            else:
                H[i, j] = vsg * np.cos(2 * np.pi * np.dot(gij, self.zinc.tau)) \
                    + 1j * vag * np.sin(2 * np.pi * np.dot(gij, self.zinc.tau))

        eigvals, eigvecs = np.linalg.eigh(H)
        energies_ev = 13.6058 * eigvals
        return energies_ev


if __name__ == '__main__':
    Si = Diamond(10.26, -0.21, 0.04, 0.08)
    Ge = Diamond(10.69, -0.23, 0.01, 0.06)
    ZnS = Zincblende(10.22, -0.22, 0.03, 0.07, 0.24, 0.14, 0.04)

    name = "Si"
    pp = Pseudopotential(Si, encut=5)


    n_ele = 4

    # L-G-X-K-G
    G = np.array([0, 0, 0])
    X = np.array([1, 0, 0])
    K = np.array([0.75, 0.75, 0])
    L = np.array([0.5, 0.5, 0.5])

    bins = 20
    path = [L, G, X, K, G]
    kpoints_line = []
    for i in range(len(path) - 1):
        line_tmp = [path[i] * (1 - j / bins) + path[i + 1] * j / bins for j in range(bins)]
        kpoints_line.extend(line_tmp)

    list_energies = []
    for kpoint in kpoints_line:
        energies = pp.solve(kpoint)
        list_energies.append(energies)

    offset = np.max([np.max(e[:n_ele]) for e in list_energies])
    emin = np.min([np.min(e[:n_ele]) for e in list_energies])

    xticks = ["" for _ in range(bins * (len(path) - 1))]
    xticks[0] = "L"
    xticks[bins] = "G"
    xticks[2 * bins] = "X"
    xticks[3 * bins] = "K"
    xticks[4 * bins - 1] = "G"

    for i, (kpoint, energies) in enumerate(zip(kpoints_line, list_energies)):
        e_plot = energies[energies < offset + (offset - emin)] - offset
        n_bands = len(e_plot)
        plt.scatter([i] * n_ele, e_plot[:n_ele], c='r')
        plt.scatter([i] * (n_bands - n_ele), e_plot[n_ele:], c='b')

    plt.title(f"Band structure of {name}")
    plt.ylabel('E [eV]')
    plt.xticks(range(len(xticks)), xticks)
    # plt.show()
    plt.savefig(f"{name}.png")
